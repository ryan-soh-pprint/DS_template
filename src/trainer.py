import json
import os
import shutil
import time
from typing import Union
import uuid

import matplotlib.pyplot
import mlflow
import optuna
import pandas as pd
from optuna.storages import RDBStorage
from pptoolbox.visualization import plot_confusion_matrix_v2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from utils import cross_group_predict
import numpy as np
import pickle
import joblib
from optuna.trial import TrialState

matplotlib.pyplot.switch_backend("Agg")

from objective import BaseClassifyObjective

class AtomicSamplerCheckpoint:
    def __init__(self, filepath="sampler_checkpoint.pkl"):
        self.filepath = filepath
        self.tempfile = f"{filepath}.tmp"

    def __call__(self, study, trial):
        if trial.state == TrialState.COMPLETE:
            # Save study ID with sampler
            data = {
                "sampler": study.sampler,
                "study_id": study._study_id  # Add study ID tracking
            }
            self._atomic_save(data)

    def _atomic_save(self, data):
        with open(self.tempfile, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(self.tempfile, self.filepath)

    def load(self, current_study_id):
        try:
            with open(self.filepath, "rb") as f:
                data = pickle.load(f)
                if data["study_id"] != current_study_id:
                    return None  # Discard mismatched checkpoints
                return data["sampler"]
        except (FileNotFoundError, KeyError):
            return None

class BaseClassifyTrainerV4:
    def __init__(
        self,
        mlflow_experiment_name: str,
        optuna_study_name: str,
        # mlflow_tracking_uri: str,
        optuna_storage_url: str,
        sampler: optuna.samplers.BaseSampler,
        n_total_trials: int,
        objective: BaseClassifyObjective,
        direction: Union[list, str],
        metric_name: list,
        seed: int,
        additional_tags: dict = {},
        sampler_checkpoint_path: str = None,  # Make path study-dependent
        enable_checkpointing: bool = True,
        encoder: LabelEncoder = None,
    ):

        self.experiment_name = mlflow_experiment_name
        self.study_name = optuna_study_name

        self.seed = seed
        storage = RDBStorage(url=optuna_storage_url)

        self.objective = objective
        self.n_total_trials = n_total_trials

        self._additional_tags = additional_tags

        self._best_pipeline = None
        self.encoder = encoder

        # Add sampler checkpointing
        # Generate study-specific checkpoint path
        if sampler_checkpoint_path is None:
            sampler_checkpoint_path = f"sampler_{optuna_study_name}.pkl"
        self.sampler_checkpoint = AtomicSamplerCheckpoint(sampler_checkpoint_path)
        self.enable_checkpointing = enable_checkpointing

        # Check study existence before loading sampler
        try:
            self._study = optuna.load_study(
                study_name=optuna_study_name,
                storage=storage
            )
            current_study_id = self._study._study_id
        except KeyError:
            # Create a new study if it doesn't exist

            if isinstance(direction, list):
                self._study = optuna.create_study(
                    study_name=optuna_study_name,
                    directions=direction,
                    storage=storage,
                    sampler=sampler,
                    load_if_exists=False,
                )

            elif isinstance(direction, str):
                self._study = optuna.create_study(
                    study_name=optuna_study_name,
                    direction=direction,
                    storage=storage,
                    sampler=sampler,
                    load_if_exists=False,
                )

            current_study_id = self._study._study_id
            if enable_checkpointing and os.path.exists(sampler_checkpoint_path):
                os.remove(sampler_checkpoint_path)  # Clear old checkpoint

        # Load sampler with study ID validation
        if enable_checkpointing:
            self.sampler = self.sampler_checkpoint.load(current_study_id) or sampler

        self._study.set_metric_names(metric_name)
        # For trial pruning
        self.objective._add_study(self._study)

        self.study_id = storage.get_study_id_from_name(self.study_name)
        # mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.experiment_id = self._get_or_create_experiment(self.experiment_name)
        self._init_mlflow_run()

    @staticmethod
    def _get_or_create_experiment(experiment_name) -> str:
        """
        Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

        This function checks if an experiment with the given name exists within MLflow.
        If it does, the function returns its ID. If not, it creates a new experiment
        with the provided name and returns its ID.

        Parameters:
        - experiment_name (str): Name of the MLflow experiment.

        Returns:
        - str: ID of the existing or newly created MLflow experiment.
        """

        if experiment := mlflow.get_experiment_by_name(experiment_name):
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name)

    def _init_mlflow_run(self):
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            self._run_id = run.info.run_id

    # def run(self, save_best_model: bool = True):
    #     # X_train = self.objective.X_train
    #     # y_train = self.objective.y_train
    #     # self._log_training_data(X_train=X_train, y_train=y_train)

    #     # Add pre-optimization cleanup
    #     if self.enable_checkpointing:
    #         self._cleanup_incomplete_trials()

    #     # Modify optimize call
    #     callbacks = []
    #     if self.enable_checkpointing:
    #         callbacks.append(self.sampler_checkpoint)

    #     # Calculate remaining trials after cleanup
    #     completed_trials = sum(1 for t in self._study.trials)
    #     remaining_trials = max(0, self.n_total_trials - completed_trials)

    #     self._study.optimize(
    #         self.objective,
    #         n_trials=remaining_trials,
    #         catch=Exception,
    #         show_progress_bar=True,
    #         callbacks=callbacks,  # Add callbacks here
    #         gc_after_trial=True
    #     )

    #     if save_best_model:
    #         best_params = self.get_best_trial_params()
    #         self.save_best_model(best_params)

    def run(self, save_best_model: bool = True):
        if self.enable_checkpointing:
            self._cleanup_incomplete_trials()

        completed_trials = sum(1 for t in self._study.trials if t.state != TrialState.FAIL)
        remaining_trials = max(0, self.n_total_trials - completed_trials)

        if remaining_trials > 0:
            callbacks = [self.sampler_checkpoint] if self.enable_checkpointing else []
            self._study.optimize(
                self.objective,
                n_trials=remaining_trials,
                catch=(Exception,),
                show_progress_bar=True,
                callbacks=callbacks,
                gc_after_trial=True
            )

        if save_best_model:
            best_params = self.get_best_trial_params()
            self.save_best_model(best_params)

    def save_best_model(self, best_params: dict):
        self._best_pipeline = self.objective.construct_pipeline_from_params(best_params)
        X_train = self.objective.X_train
        y_train = self.objective.y_train
        X_test = self.objective.X_test
        y_test = self.objective.y_test

        self._best_pipeline.fit(X_train, y_train)

        with mlflow.start_run(run_id=self._run_id):
            if self.enable_checkpointing:
                mlflow.log_artifact(self.sampler_checkpoint.filepath)
                mlflow.log_param("sampler_state", 
                    os.path.basename(self.sampler_checkpoint.filepath))
            with mlflow.start_run(experiment_id=self.experiment_id, nested=True):
                mlflow.log_params(best_params)
                mlflow.sklearn.log_model(self._best_pipeline, "model")

                """
                save deployment pipeline directly to pkl file
                """
                # preparing the pipeline for platform
                imba_samplers = ["ClusterCentroids","RandomUnderSampler","TomekLinks","RandomOverSampler","SMOTE","ADASYN","SMOTEENN","SMOTETomek"]
                filtered_steps = [(name, step) for name, step in self._best_pipeline.steps if name not in imba_samplers]
                platform_pipeline = Pipeline(steps=filtered_steps)
                # Save the modified pipeline
                with open("origin_model_prediction.pkl", "wb") as f:
                    pickle.dump(platform_pipeline, f)
                mlflow.log_artifact("origin_model_prediction.pkl", artifact_path="deployment_pipeline")

                logged_tags = {
                    "project": "classify_imba",
                    "optimizer_engine": "optuna",
                    "search_space": "imba",
                    "random_state": self.seed,
                    "model": best_params["model"],
                    "optuna_study_id": self.study_id,
                }
                logged_tags.update(self._additional_tags)
                mlflow.set_tags(logged_tags)

                cv = StratifiedGroupKFold(n_splits=3)
                cv_preds = cross_group_predict(
                    self._best_pipeline,
                    X_train,
                    y_train,
                    cv,
                    groups=X_train.index,
                )
                fig, _ = plot_confusion_matrix_v2(
                    y_train.groupby(y_train.index).mean(),
                    cv_preds,
                    class_labels=self.encoder.classes_,
                    figsize=(5, 5),
                    title="CV | " + self._score_results(y_train.groupby(y_train.index).mean(), cv_preds),
                )
                mlflow.log_figure(fig, "confusion_matrix_train.png")
                cv_bal_acc = balanced_accuracy_score(
                    y_train.groupby(y_train.index).mean(), cv_preds
                )
                mlflow.log_metric("cv_balanced_accuracy", cv_bal_acc)

                y_pred_lot = self._best_pipeline.predict(X_test.reset_index().groupby("lot_id").mean())
                fig, _ = plot_confusion_matrix_v2(
                    y_test.groupby(y_test.index).mean(),
                    y_pred_lot,
                    class_labels=self.encoder.classes_,
                    figsize=(5, 5),
                    title="Test | " + self._score_results(y_test.groupby(y_test.index).mean(), y_pred_lot),
                )
                mlflow.log_figure(fig, "confusion_matrix_test_lot.png")

                def classify_mode_pred(pipeline, X):
                    pred = pd.Series(pipeline.predict(X), index = X.index)

                    return pred.groupby(pred.index).apply(lambda x: x.value_counts().idxmax())
                
                y_pred_mode = classify_mode_pred(self._best_pipeline, X_test)
                fig, _ = plot_confusion_matrix_v2(
                    y_test.groupby(y_test.index).mean(),
                    y_pred_mode,
                    class_labels=self.encoder.classes_,
                    figsize=(5, 5),
                    title="Test | " + self._score_results(y_test.groupby(y_test.index).mean(), y_pred_mode),
                )
                mlflow.log_figure(fig, "confusion_matrix_test_mode.png")

    def _cleanup_incomplete_trials(self):
        """Mark incomplete trials as FAILED using Optuna's trial system."""
        study = self._study
        storage = study._storage
        
        for trial in study.get_trials(states=(TrialState.RUNNING, TrialState.WAITING)):
            try:
                storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
            except Exception as e:
                print(f"Failed to update trial {trial._trial_id}: {e}")


    # def _log_training_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    #     with mlflow.start_run(run_id=self._run_id):
    #         now = int(time.time())
    #         tmp_dir = f"tmp_{now}"
    #         if not os.path.exists(tmp_dir):
    #             os.makedirs(tmp_dir)
    #         X_train.to_csv(f"{tmp_dir}/X_train.csv")
    #         y_train.to_csv(f"{tmp_dir}/y_test.csv")
    #         mlflow.log_artifact(f"{tmp_dir}/X_train.csv", artifact_path="data")
    #         mlflow.log_artifact(f"{tmp_dir}/y_test.csv", artifact_path="data")
    #         shutil.rmtree(tmp_dir)

    def get_best_trial_params(self) -> dict:
        return self._study.best_params  # For single-objective optimization

    @staticmethod
    def _score_results(y_true, y_pred) -> str:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return f"Accuracy: {acc:.2f} | Balanced Accuracy: {bal_acc:.2f} | F1: {f1:.2f}"

    def get_study(self):
        return self._study
    
class MultiObjClassifyTrainerV4(BaseClassifyTrainerV4):
    def get_best_trial_params(self):
        best_trials = self._study.best_trials  # because multi-objective has BEST trials
        results = np.array([trial.values for trial in best_trials])
        results = pd.DataFrame(results).sort_values([0, 1], ascending=False)  # sort by 2 features
        best_trial_idx = results.index[0]
        best_params = best_trials[best_trial_idx].params
        return best_params