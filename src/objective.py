import pandas as pd
import optuna
import time

from optuna import Trial, Study
from sklearn.decomposition import PCA
from sklearn.base import clone

from pympler import asizeof
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from utils import cross_group_validate

from search_space import DIM_RED_MODELS

from search_space import (
    construct_classifier_step,
    construct_classifier_step_from_params,
    construct_dim_red_step,
    construct_dim_red_step_from_params,
    construct_imbal_sampler_step,
    construct_imbal_sampler_step_from_params,
    construct_preprocessor,
    construct_preprocessor_from_params,
)

BINARY_METRICS = ["balanced_accuracy", "f1_macro", "precision_macro", "recall_macro"]
BINARY_CRITERION = "balanced_accuracy"

class BaseClassifyObjective:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
    ):
        assert X_train.index.name == "lot_id"
        assert y_train.index.name == "lot_id"
        assert X_test.index.name == "lot_id"
        assert y_test.index.name == "lot_id"

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def construct_pipeline_from_trial(self, trial: Trial) -> Pipeline:
        raise NotImplementedError

    def construct_pipeline_from_params(self, params: dict) -> Pipeline:
        raise NotImplementedError

    def _prune_repeated_trials(self, current_trial: Trial) -> None:
        trials = self._study.get_trials(deepcopy=False)
        current_params = current_trial.params

        for t in trials:
            if t.state.is_finished() and t.params == current_params:
                raise optuna.TrialPruned(f"Pruning trial {current_trial.number} due to duplicate parameters.")

    def _add_study(self, study: Study) -> None:
        self._study = study


class DefaultClassifyV4(BaseClassifyObjective):
    def __call__(self, trial: Trial):
        pipeline = self.construct_pipeline_from_trial(trial)
        self._prune_repeated_trials(trial)
        pipeline.fit(self.X_train, self.y_train)
        # pipeline_size = asizeof.asizeof(pipeline)

        cv = StratifiedGroupKFold(n_splits=3)
        cv_results, cv_preds = cross_group_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv,
            groups=self.X_train.index,
            scoring=BINARY_METRICS,
        )
        
        performance = cv_results[f"mean_test_{BINARY_CRITERION}"]
        # trial.set_user_attr("pipeline_size", pipeline_size)
        trial.set_user_attr("performance", performance)

        return performance

    def construct_pipeline_from_trial(self, trial: Trial) -> Pipeline:
        pipeline_steps = construct_preprocessor(trial)
        classifier = construct_classifier_step(trial)
        dim_red = (
            True
            if classifier[0] in DIM_RED_MODELS
            else trial.suggest_categorical("dim_red", [False, True])
        )
        if dim_red:
            dim_red_step = construct_dim_red_step(trial)
            pipeline_steps.append(dim_red_step)
        pipeline_steps.append(classifier)
        return clone(Pipeline(pipeline_steps))

    def construct_pipeline_from_params(self, params: dict) -> Pipeline:
        pipeline_steps = construct_preprocessor_from_params(params)
        classifier = construct_classifier_step_from_params(params)
        dim_red = params.get("dim_red", False) or classifier[0] in DIM_RED_MODELS
        if dim_red:
            dim_red_step = construct_dim_red_step_from_params(params)
            pipeline_steps.append(dim_red_step)
        pipeline_steps.append(classifier)
        return clone(Pipeline(pipeline_steps))
    
class MultiObjectiveClassifyV4(DefaultClassifyV4):
    def __call__(self, trial: Trial):
        pipeline = self.construct_pipeline_from_trial(trial)
        self._prune_repeated_trials(trial)
        pipeline.fit(self.X_train, self.y_train)
        # pipeline_size = asizeof.asizeof(pipeline)

        cv = StratifiedGroupKFold(n_splits=3)
        cv_results, cv_preds = cross_group_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv,
            groups=self.X_train.index,
            scoring=BINARY_METRICS,
        )

        f1_macro = f1_score(self.y_train.groupby(self.y_train.index).mean(), cv_preds, average="macro")
        tn, fp, fn, tp = confusion_matrix(self.y_train.groupby(self.y_train.index).mean(), cv_preds).ravel()
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return f1_macro, recall, specificity
    
class ImbaClassifyV4(BaseClassifyObjective):
    def __call__(self, trial: Trial):
        pipeline = self.construct_pipeline_from_trial(trial)
        self._prune_repeated_trials(trial)
        pipeline.fit(self.X_train, self.y_train)
        # pipeline_size = asizeof.asizeof(pipeline)

        cv = StratifiedGroupKFold(n_splits=3)
        cv_results, cv_preds = cross_group_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv,
            groups=self.X_train.index,
            scoring=BINARY_METRICS,
        )
        
        performance = cv_results[f"mean_test_{BINARY_CRITERION}"]
        # trial.set_user_attr("pipeline_size", pipeline_size)
        trial.set_user_attr("performance", performance)

        return performance

    def construct_pipeline_from_trial(self, trial: Trial) -> Pipeline:
        pipeline_steps = construct_preprocessor(trial)
        classifier = construct_classifier_step(trial)
        sampler = construct_imbal_sampler_step(trial)
        if sampler is not None:
            pipeline_steps.append(sampler)
        dim_red = (
            True
            if classifier[0] in DIM_RED_MODELS
            else trial.suggest_categorical("dim_red", [False, True])
        )
        if dim_red:
            dim_red_step = construct_dim_red_step(trial)
            pipeline_steps.append(dim_red_step)
        pipeline_steps.append(classifier)
        if sampler is not None:
            return clone(ImbPipeline(pipeline_steps))
        return clone(Pipeline(pipeline_steps))


    def construct_pipeline_from_params(self, params: dict) -> Pipeline:
        pipeline_steps = construct_preprocessor_from_params(params)
        classifier = construct_classifier_step_from_params(params)
        sampler = construct_imbal_sampler_step_from_params(params)
        if sampler is not None:
            pipeline_steps.append(sampler)
        dim_red = params.get("dim_red", False) or classifier[0] in DIM_RED_MODELS
        if dim_red:
            dim_red_step = construct_dim_red_step_from_params(params)
            pipeline_steps.append(dim_red_step)
        pipeline_steps.append(classifier)
        if sampler is not None:
            return clone(ImbPipeline(pipeline_steps))
        return clone(Pipeline(pipeline_steps))
    
class MultiObjectiveImbaClassify(ImbaClassifyV4):
    def __call__(self, trial: Trial):
        pipeline = self.construct_pipeline_from_trial(trial)
        self._prune_repeated_trials(trial)
        pipeline.fit(self.X_train, self.y_train)
        # pipeline_size = asizeof.asizeof(pipeline)

        cv = StratifiedGroupKFold(n_splits=3)
        cv_results, cv_preds = cross_group_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv,
            groups=self.X_train.index,
            scoring=BINARY_METRICS,
        )

        # calulate minority recall and majority precision

        minority_class = self.y_train.value_counts().idxmin()
        majority_class = self.y_train.value_counts().idxmax()

        minority_recall = recall_score(
            self.y_train.groupby(self.y_train.index).mean(), cv_preds, pos_label=minority_class
        )
        majority_recall = recall_score(
            self.y_train.groupby(self.y_train.index).mean(), cv_preds, pos_label=majority_class
        )

        threshold = 0.5
        if majority_recall < threshold:
            raise optuna.TrialPruned(f'Pruning trial {trial.number} because majority recall = {majority_recall} < {threshold}')

        balanced_accuracy = balanced_accuracy_score(
            self.y_train.groupby(self.y_train.index).mean(), cv_preds
        )
        return minority_recall, balanced_accuracy
    
class CustomObjectiveImbaClassify(ImbaClassifyV4):
    def __call__(self, trial: Trial):
        pipeline = self.construct_pipeline_from_trial(trial)
        self._prune_repeated_trials(trial)
        pipeline.fit(self.X_train, self.y_train)
        # pipeline_size = asizeof.asizeof(pipeline)

        cv = StratifiedGroupKFold(n_splits=3)
        cv_results, cv_preds = cross_group_validate(
            pipeline,
            self.X_train,
            self.y_train,
            cv,
            groups=self.X_train.index,
            scoring=BINARY_METRICS,
        )

        # calulate minority recall and majority precision

        minority_class = self.y_train.value_counts().idxmin()
        majority_class = self.y_train.value_counts().idxmax()

        minority_recall = recall_score(
            self.y_train.groupby(self.y_train.index).mean(), cv_preds, pos_label=minority_class
        )
        majority_recall = recall_score(
            self.y_train.groupby(self.y_train.index).mean(), cv_preds, pos_label=majority_class
        )

        balanced_accuracy = balanced_accuracy_score(
            self.y_train.groupby(self.y_train.index).mean(), cv_preds
        )

        # prune if not within threshold
        threshold = 0.7
        if majority_recall < threshold:
            raise optuna.TrialPruned(f'Pruning trial {trial.number} because majority recall = {majority_recall} < {threshold}')
        if balanced_accuracy == 1:
            raise optuna.TrialPruned(f'Pruning trial {trial.number} because bal_acc = {balanced_accuracy} means overfitting')
        
        performance = cv_results[f"mean_test_{BINARY_CRITERION}"]
        # trial.set_user_attr("pipeline_size", pipeline_size)
        trial.set_user_attr("minority_recall", minority_recall)

        return minority_recall