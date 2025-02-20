from collections.abc import Callable
from typing_extensions import Self
from typing import Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from hyperopt.pyll.base import SymbolTable

import pandas as pd
import numpy as np
import os
import pickle as pkl

from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from functools import partial
from os.path import join
from time import time

from hyperopt import STATUS_OK, STATUS_FAIL, Trials, hp, fmin, tpe, space_eval
from hyperopt.pyll.stochastic import sample
from hyperopt.early_stop import no_progress_loss

from utils_custom import * #changed
from pipelines_custom import * # change, added choice of no preprocessing
# from pptoolbox.platform.automl_v4.pipelines import *
from pptoolbox.platform.automl_v4.exceptions import AutoMLException

import warnings
warnings.simplefilter("ignore")


class AutoML_v4_custom(BaseEstimator): ##changed line 478 create_custom_pipeline
    USEFUL_TRIAL_INFO = ["tid", "result", "status"]
    DEFAULT_EXPLORATION = 50
    DEFAULT_EXPLOITATION = 150
    DEFAULT_STAGNANT = 75
    RANDOM_STATE = 42
    DEFAULT_RSTATE = np.random.default_rng(RANDOM_STATE)

    CLASSIFY_CUTOFF = 20

    CRITERIA_MAP = {"profile": "test_neg_mean_squared_error",
                    "classify": "test_balanced_accuracy"}
    KFOLD_MAP = {"profile": KFOLD, "classify": STRATIFIED_KFOLD}
    SEARCH_SPACE_MAP = {"profile": REGRESSION_PIPELINES,
                        "classify": CLASSIFIER_PIPELINES}

    SAMPLE_UNNEEDED = ["regressor", "classifier",
                       "dimred_choice", "preprocessing"]

    @staticmethod
    def simplify_trial(trial: dict) -> dict:
        """Remove useless information from the hyperopt trials dictionary
        for simpler examination. Useful information is defined in the class constant
        `USEFUL_TRIAL_INFO`

        Inputs
            trial: Hyperopt trial dictionary

        Output
            Simplified dictionary with useful information
        """

        unpacked_sample = AutoML_v4_custom.unpack_sample(trial)
        useful_info = {
            key: value
            for key, value in trial.items()
            if key in AutoML_v4_custom.USEFUL_TRIAL_INFO
        }

        useful_info["sample"] = unpacked_sample
        return useful_info

    @staticmethod
    def unpack_sample(
        trial: dict,
        choices_params: dict = CHOICES,     # from __init__
    ) -> dict:
        """Hyperopt sampling keeps the sampled values in a list.
        This helper method unpacks this list into an easier to work with dictionary
        to examine the search space sample

        Inputs
            trial: Hyperopt trial dictionary

            choices_params: Choices defined using hp.choice. By default, only
                window and kernel search spaces are defined by hp.choice. If more
                choices are added into the search space, this choices_params needs
                to be given by the caller

        Output
            Dictionary with the unpacked parameters
        """
        sample = trial["misc"]["vals"]

        unpacked = {}
        for key, value in sample.items():
            if len(value) == 0:
                continue

            if key in AutoML_v4_custom.SAMPLE_UNNEEDED:
                continue

            if key in CHOICES:
                unpacked_value = CHOICES[key][value[0]]
            else:
                unpacked_value = value[0]
            unpacked[key] = unpacked_value

        return unpacked

    @staticmethod
    def adjust_kfold(
        kfold: BaseCrossValidator, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> BaseCrossValidator:
        """
        Ensure that for models with internal cross validation, training data passed
        to it will have enough data to split during internal cross validation (Only for classify tasks)

        Input
            kfold: Cross validation object
            X: CV Input
            y: CV labels

        Output
            Adjusted kfold with the appropriate number of CV splits
        """

        num_samples = len(X)
        num_classes = len(np.unique(y))

        # num_samples * ((cv_splits - 1) / cv_splits) * 0.1 > num_classes
        # (cv_splits - 1) / cv_splits > (10 * num_classes) / num_samples
        # 1 - (1 / cv_splits) > (10 * num_classes) / num_samples
        # 1 / cv_splits < 1 - ((10 * num_classes) / num_samples)
        # 1 / cv_splits < ((num_samples) - (10 * num_classes)) / num_samples
        # cv_splits > num_samples / ((10 * num_classes) - num_samples)

        # If the dataset does not have enough data for internal CV, raise a warning straight away
        if (num_samples * 0.1) <= num_classes:
            raise AutoMLException(
                "Dataset is too small for cross validation training. "
                "Please increase the number of samples"
            )

        num_splits = int(num_samples / (num_samples - (10 * num_classes)))

        if num_splits <= 1:
            raise AutoMLException(
                "Dataset is too small for cross validation training. "
                "Please increase the number of samples"
            )

        if kfold.n_splits > num_splits:
            return kfold

        kfold.n_splits = num_splits
        return kfold

    def __init__(
        self,
        task_type: str = None,
        criteria: str = None,
        scoring: list = None,
        loss_fn: Callable = None,
        stop_fn: Callable = None,
        exploration_runs: int = DEFAULT_EXPLORATION,
        exploitation_runs: int = DEFAULT_EXPLOITATION,
        stagnant_runs: int = DEFAULT_STAGNANT,
        rstate: np.random.default_rng = DEFAULT_RSTATE,
        log: bool = False,
    ) -> None:
        """Initialises the AutoML v4 Trainer

        Inputs
            task_type: Either 'profile' or 'classify'. If None, will be inferred at fit time

            criteria: Metric name used to define the loss value of a pipeline. If None,
                defaults to validation MSE for profile tasks, and validation 
                balanced accuracy for classify tasks.

                Should be a sklearn metric name prefixed by 'train' for train metrics,
                and 'test' for cross validation metrics (sklearn cross validation naming convention)

                Eg. 'test_neg_mean_squared_error' to use cross validation MSE as the loss value

            scoring: List of metric names to be calculated during optimisation process. 
                If None, will be defaulted to 
                binary classification - balanced accuracy
                multiclass classification - balanced accuracy, precision
                regression - r2, MSE and MAE

            stop_fn: A function that checks if an early stopping criteria has been met.
                The stop function must be a callable with signature:
                    stop_fn(hyperopt_trial: dict, **kwargs) -> (bool, kwargs)

                The boolean return value specifies whether or not to stop the 
                optimisation process.
                Additional variables can be passed into the stop function using **kwargs
                (See stopping_criteria)
                If None, defaults to stopping criteria with default exploration, exploitation
                and stagnant

            exploration_runs: Number of random runs to perform during TPE sampling. If None,
                defaults to 50 runs

            exploitation_runs: Number of exploitation runs. If None, defaults to
                150 runs

            stagnant_runs: Maximum number of allowable EXPLOITATION runs without 
                improvement before stopping. If None, defaults to 75 runs.
                See Confluence - Hyperopt Kitchen Sink for more details

            rstate: Random number generator for reproduceability

            log: Whether or not to log the optimistion process onto MLFlow
        """

        if task_type and (task_type.lower() not in ["profile", "classify"]):
            raise AutoMLException(
                "`task type` can only be 'profile' or 'classify'")

        self.__task_type = task_type
        self.__fitted = False
        self.__pipeline = None
        self.__trials = None

        if criteria:
            # Check if criteria is prefixed by train/test
            prefix = criteria.split("_")[0]
            if prefix not in ["train", "test"]:
                raise AutoMLException(
                    "Criteria must be a sklearn metric name prefixed by "
                    "'train' or 'test'"
                )
            
            # Check if criteria given is a valid sklearn metric
            all_scorer_names = get_scorer_names()
            stripped_criteria = (
                criteria
                .lower()
                .replace("train_", "")
                .replace("test_", "")
            )
            if stripped_criteria not in all_scorer_names:
                raise AutoMLException(
                    f"Invalid criteria {criteria}. Criteria is not a valid sklearn metric"
                )

        if scoring:
            all_scorer_names = get_scorer_names()
            for score_name in scoring:
                if score_name in all_scorer_names:
                    continue

                raise AutoMLException(
                    f"Invalid score name {score_name}. Scores must be a sklearn metric name"
                )

        self.__criteria = criteria
        self.__scoring = scoring
        self.__loss_fn = loss_fn
        self.__stop_fn = stop_fn

        self.__exploration_runs = exploration_runs
        self.__exploitation_runs = exploitation_runs
        self.__stagnant_runs = stagnant_runs
        self.__rstate = rstate
        self.__log = log
        self.__log_config = None
        self.__kfold = None
        self.__encoder = None


    def set_log_config(
        self,
        experiment_name: str,
        log_metrics: bool = True,
        log_time: bool = True,
        log_loss: bool = True,
        log_hyperparams: bool = True,
        mlflow_uri: str = None,
    ) -> None:
        """Sets up the logging configuration for MLFlow tracking

        Inputs
            experiment_name: Experiment name for subsequent runs to be grouped under
            log_metrics: Whether or not to log metrics information
            log_time: Whether or not to log time information
            log_loss: Whether or not to log loss information
            log_hyperparams: Whether or not to log hyperparams information
            mlflow_uri: Link to an mlflow server if needed
        """

        if mlflow_uri is not None:
            mlflow.set_tracking_uri(mlflow_uri)

        exp_details = mlflow.get_experiment_by_name(experiment_name)
        if exp_details is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = dict(exp_details)["experiment_id"]

        self.__log_config = {
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "log_metrics": log_metrics,
            "log_time": log_time,
            "log_loss": log_loss,
            "log_hyperparams": log_hyperparams,
        }
        return

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        search_space: SymbolTable = None,
        kfold: BaseCrossValidator = None,
        verbose: bool = True,
        **kwargs
    ) -> Self:
        """Performs Bayesian Optimisation with TPE

        Inputs
            X_train, y_train: Spectra and Labels. The scanning groups (lot_name, lot_id)
                of the data must be in the index for splitting of data by scanning groups.
                Can also have groups be passed in as an additional argument (Future updates),
                but its cleaner to do it this way

            search_space: The search space for the current optimisation process. If None,
                defaults to the pipelines defined in pipelines.py

            kfold: Cross validator object. If None, GroupKFold will be used for profile
                tasks, and StratifiedGroupKFold will be used for classify tasks

            verbose: Whether or not to display progress bar

            **kwargs passed to loss function
        """
        # Ensure that X and y are appropriate types
        if not isinstance(X_train, pd.DataFrame):
            raise AutoMLException(
                "Input X must be a pandas Dataframe"
            )

        if not isinstance(y_train, pd.Series):
            raise AutoMLException(
                "Input y must be a pandas Series"
            )

        if not all(X_train.index == y_train.index):
            raise AutoMLException(
                "Index of X and y must be aligned"
            )

        # Check if the user has provided X and y without grouping indices
        if all(X_train.index == pd.RangeIndex(len(X_train))):
            raise AutoMLException(
                "Ensure X indices are set to grouping columns"
            )

        if all(y_train.index == pd.RangeIndex(len(y_train))):
            raise AutoMLException(
                "Ensure y indices are set to grouping columns"
            )

        # Ensure that the labels provided by the user has been encoded for classify tasks.
        # Store the encoder for the user to encode the test set
        encoder, y_train = ensure_encoded_y(y_train)
        if encoder is not None:
            self.__encoder = encoder

        # If the user has not specified parameters for the optimisation
        # set them to the default values during fit

        # Since default parameters depends heavily on task_type, we settle task_type first
        num_unique_y = len(np.unique(y_train))
        if self.__task_type is None:
            if num_unique_y > self.CLASSIFY_CUTOFF:
                self.__task_type = "profile"
            else:
                self.__task_type = "classify"

        # Retrieve criteria from default criteria map
        if self.__criteria is None:
            self.__criteria = self.CRITERIA_MAP[self.__task_type]

        # Retrieve default scoring metrics from pipelines.py
        if self.__scoring is None:
            if (self.__task_type == "classify") and (num_unique_y == 2):
                self.__scoring = BINARY_METRICS
            elif (self.__task_type == "classify") and (num_unique_y > 2):
                self.__scoring = MULTICLASS_METRICS
            else:
                self.__scoring = REGRESSION_METRICS

        # If the given criteria is not inside our list of scorers, add it in
        if self.__criteria not in self.__scoring:
            stripped_criteria = (
                self.__criteria
                .lower()
                .replace("train_", "")
                .replace("test_", "")
            )
            self.__scoring.append(stripped_criteria)

        # Retrieve default search space from pipelines.py
        if search_space is None:
            search_space = self.SEARCH_SPACE_MAP[self.__task_type]

        # Retrieve default cross validation method from pipelines.py
        if kfold is None:
            kfold = self.KFOLD_MAP[self.__task_type]

        # if self.__task_type == "classify":
        #     kfold = self.adjust_kfold(kfold, X_train, y_train)
        # self.__kfold = kfold

        # Default stop function cannot be set as a class attribute,
        # as it needs to be recreated anytime the run values are changed using set_params
        # But if the user gives their own stop function, we accept that as well
        if self.__stop_fn is None:
            stop_fn = (
                stopping_criteria(
                    self.__exploration_runs,
                    self.__exploitation_runs,
                    self.__stagnant_runs
                )
                if self.__stagnant_runs > 0
                else None
            )
        else:
            stop_fn = self.__stop_fn

        if self.__loss_fn is None:
            if not self.__log:
                self.__loss_fn = loss_function
            else:
                self.__loss_fn = loss_function_w_logger

        if (self.__log) and (self.__log_config is None):
            raise AutoMLException(
                "Logging config for MLFLow has not been set. " +
                "Set the logging config by calling AutoML_v4_custom.set_log_config()"
            )

        self.__trials = Trials()
        f = partial(
            self.__loss_fn,
            X_train=X_train,
            y_train=y_train,
            scoring=self.__scoring,
            criteria=self.__criteria,
            kfold=kfold,
            log_config=self.__log_config,
            **kwargs
        )
        optimised = fmin(
            fn=f,
            space=search_space,
            trials=self.__trials,
            algo=partial(
                tpe.suggest,
                n_startup_jobs=self.__exploration_runs
            ),
            max_evals=self.__exploitation_runs + self.__exploration_runs,
            early_stop_fn=stop_fn,
            rstate=self.__rstate,
            verbose=verbose,
        )

        best_pipeline_sample = space_eval(search_space, optimised)
        best_pipeline, _ = create_custom_pipeline(best_pipeline_sample) ##changed!!
        print(best_pipeline)
        best_pipeline.fit(X_train, y_train)

        self.__fitted = True
        self.__pipeline = best_pipeline

        return self

    def check_fitted(self) -> bool:
        if not self.__fitted:
            raise AutoMLException("AutoML is not fitted yet")

        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Runs prediction on a given spectral input

        Inputs
            X: Self explanatory

        Outputs
            Prediction on X, using model selected from optimisation
        """

        self.check_fitted()
        return self.__pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get predicted probabilities only for classification tasks

        Inputs
            X: Duh

        Outputs
            Predicted probabilities for each class, using model selected from optimisation
        """

        self.check_fitted()
        if self.__task_type != "classify":
            raise AutoMLException(
                "predict_proba is only used for classify tasks")
        return self.__pipeline.predict_proba(X)

    def get_trials(self) -> list:
        """Returns a list containing the trials sampled during optimisation

        Outputs
            Hyperopt Trials object
        """

        self.check_fitted()
        return self.__trials.trials.copy()

    def get_best_trial(self) -> dict:
        """Returns the trial with the smallest loss value

        Outputs
            Dictionary with the sampled parameters and results of evalaution
        """

        self.check_fitted()
        return self.__trials.best_trial.copy()

    def get_best_performance(self) -> float:
        """Returns the performance of the best pipeline from the optimisation process

        Outputs
            Highest performance value
        """

        self.check_fitted()
        return self.__trials.best_trial["result"]["cv_results"][self.__criteria]

    def get_best_loss(self) -> float:
        """Returns the smallest loss value from the optimisation process

        Outputs
            Smallest optimisation loss
        """

        self.check_fitted()
        return self.__trials.best_trial["result"]["loss"]

    def get_pipeline(self) -> Pipeline:
        """Returns the pipeline chosen by TPE

        Outputs
            sklearn Pipeline
        """

        self.check_fitted()
        return self.__pipeline

    def get_pipeline_name(self) -> str:
        """Returns the name of the pipeline chosen by TPE

        Outputs
            String of the pipeline name
        """

        self.check_fitted()
        return "-".join([name for name, comp in self.__pipeline.steps])

    def get_training_time(self) -> float:
        """Returns the total time taken to optimise

        Outputs
            Time taken (seconds)
        """

        self.check_fitted()
        return sum([trial["result"]["time_taken"] for trial in self.get_trials()])

    def get_encoder(self) -> LabelEncoder:
        """Returns the encoder used to encode y labels during fit

        Outputs
            LabelEncoder (fitted)
        """

        self.check_fitted()
        return self.__encoder

    def get_params(self, deep: bool = True) -> dict:
        """Get the parameters of this AutoML trainer

        Outputs
            Dictionary containing parameters for the AutoML trainer
        """

        # Since the stop function is not stored as a class attribute, we create it
        # from the parameters first.
        if self.__stop_fn is None:
            stop_fn = (
                stopping_criteria(
                    self.__exploration_runs,
                    self.__exploitation_runs,
                    self.__stagnant_runs
                )
                if self.__stagnant_runs > 0
                else None
            )
        else:
            stop_fn = self.__stop_fn

        return {
            "task_type": self.__task_type,
            "criteria": self.__criteria,
            "scoring": self.__scoring,
            "loss_fn": self.__loss_fn,
            "stop_fn": stop_fn,
            "kfold": self.__kfold,
            "exploration_runs": self.__exploration_runs,
            "exploitation_runs": self.__exploitation_runs,
            "stagnant_runs": self.__stagnant_runs,
            "rstate": self.__rstate,
            "log": self.__log,
            "log_config": self.__log_config,
            "encoder": self.__encoder,
        }

    def set_params(self, **kwargs) -> Self:
        """Sets the params of the estimator post init
        """

        allowed_params = self._get_param_names()
        for param_name, param_value in kwargs.items():
            if param_name not in allowed_params:
                raise AutoMLException(
                    f"{param_name} is not accepted by AutoML v4. " +
                    f"Allowed parameters are {allowed_params}"
                )

            if param_name == "task_type":
                self.__task_type = param_value
            elif param_name == "criteria":
                self.__criteria = param_value
            elif param_name == "scoring":
                self.__scoring = param_value
            elif param_name == "loss_fn":
                self.__loss_fn = param_value
            elif param_name == "stop_fn":
                self.__stop_fn = param_value
            elif param_name == "exploration_runs":
                self.__exploration_runs = param_value
            elif param_name == "exploitation_runs":
                self.__exploitation_runs = param_value
            elif param_name == "stagnant_runs":
                self.__stagnant_runs = param_value
            elif param_name == "rstate":
                self.__rstate = param_value
            elif param_name == "log":
                self.__log = param_value
            elif param_name == "log_config":
                self.__log_config = param_value

    def __repr__(self) -> str:
        if not self.__fitted:
            return "AutoML v4: Not Fitted"

        pipeline_name = self.get_pipeline_name()
        return f"AutoML v4: {pipeline_name}"

    def __sklearn_is_fitted__(self) -> bool:
        return self.__fitted
