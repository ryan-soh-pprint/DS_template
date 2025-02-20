from sklearn.model_selection import BaseCrossValidator

import pandas as pd
import numpy as np
import mlflow

from math import ceil
from time import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import *
from sklearn.base import clone
from pandas.api.types import is_object_dtype
from pptoolbox.preprocessing import FeatureMask

from hyperopt import STATUS_OK, STATUS_FAIL

from pptoolbox.platform.automl_v4.exceptions import AutoMLException

import warnings
warnings.simplefilter("ignore")


def ensure_encoded_y(y: pd.Series) -> pd.Series:
    if not is_object_dtype(y):
        return (None, y)

    encoder = LabelEncoder()
    y_encoded = pd.Series(
        encoder.fit_transform(y), index = y.index
    )

    return encoder, y_encoded


def create_pipeline(sample: dict) -> Pipeline:
    """
    Creates a pipeline from the search space sampled by hyperopt

    Inputs
        sample: Dictionary containing the sampled parameters from the search space
    Outputs
        Pipeline containing the components and parameters sampled by hyperopt
    """
    pipeline = Pipeline([])
    hyperparams = {}
    for component_type in ["preprocessing", "dim_red"]:
        component_name, component, params = sample[component_type]
        if component is None:
            continue

        if isinstance(component, tuple):
            for comp_name, comp, param in zip(component_name, component, params):
                unpacked_params = {
                    param_name: param_value.item()
                        if isinstance(param_value, np.ndarray) 
                        else param_value
                    for param_name, param_value in param.items()
                }
                hyperparams.update(unpacked_params)

                comp.set_params(**unpacked_params)
                pipeline.steps.append((comp_name, comp))

        else:
            unpacked_params = {
                param_name: param_value.item()
                    if isinstance(param_value, np.ndarray) 
                    else param_value
                for param_name, param_value in params.items()
            }
            hyperparams.update(unpacked_params)

            component.set_params(**unpacked_params)
            pipeline.steps.append((component_name, component))

        # Add in scaler after preprocessing
        if component_type == "preprocessing":
            pipeline.steps.append(("Scaler", StandardScaler(with_std = False)))

    model_name, model, model_params = sample["model"]
    hyperparams.update(model_params)
    model.set_params(**model_params)

    pipeline.steps.append((model_name, model))
    return clone(pipeline), hyperparams

def loss_function(pipeline_comps: dict, **kwargs) -> dict:
    """
    Defines the loss function used to evaluate the pipeline sampled by hyperopt.
    Loss is defined as cross validated loss on the training data

    Results are returned as a dictionary to be stored in hyperopt's Trial object
    Inputs
        pipeline_comps: The sampled search space
        **kwargs: Additional parameters to be used in evaluation of the pipeline
    Outputs
        Dictionary containing the loss of the current trial and pipeline used in the current
        trial. 
    """
    
    X_train, y_train = kwargs["X_train"], kwargs["y_train"]
    scoring = kwargs["scoring"]
    criteria = kwargs["criteria"]
    kfold = kwargs["kfold"]

    pipeline, raw_params = create_pipeline(pipeline_comps)

    # Check if PCA has n_components greater than the number of samples provided for training
    # If so, return infinity as the loss value
    pca = None
    for name, comp in pipeline.steps:
        if name != "PCA":
            continue
        pca = comp

    # Number of samples to minimally have is the grouped dataset's amount, 
    # not the sample based one
    fold_max_n = len(X_train.groupby(X_train.index)) // kfold.n_splits

    if pca and (pca.n_components >= fold_max_n):
        cv_results_agg = {}
        
        return {
            "loss": np.inf,
            "cv_results": cv_results_agg,
            "time_taken": 0,
            "pipeline": pipeline,
            "status": STATUS_OK,
            "exception": None
        }

    try:
        start = time()
        cv_results = cross_group_validate(
            estimator = pipeline,
            X = X_train,
            y = y_train,
            cv = kfold,
            groups = X_train.index,
            scoring = scoring,
        )
        time_taken = time() - start

        # Aggregate results across all folds
        cv_results_agg = {
            name: np.mean(value)
            for name, value in cv_results.items()
        }
        
        performance = cv_results_agg[criteria]
        if "neg" in criteria:
            loss = -performance
        elif criteria in ["r2", "max_error", "explained_variance", "d2_absolute_error_score"]:
            loss = performance
        else:
            loss = 1 - performance
        
        pipeline_parameters = pipeline.get_params()
        copied_pipeline = clone(pipeline)
        copied_pipeline.set_params(**pipeline_parameters)
    
        return {
            "loss": loss,
            "cv_results": cv_results_agg,
            "time_taken": time_taken,
            "pipeline": copied_pipeline,
            "status": STATUS_OK,
            "exception": None
        }

    except Exception as e:
        cv_results_agg = {}
        return {
            "loss": np.inf,
            "cv_results": cv_results_agg,
            "time_taken": 0,
            "pipeline": pipeline,
            "status": STATUS_OK,
            "exception": e
        }


def loss_function_w_logger(pipeline_comps, **kwargs):
    X_train, y_train = kwargs["X_train"], kwargs["y_train"]
    scoring = kwargs["scoring"]
    criteria = kwargs["criteria"]
    kfold = kwargs["kfold"]
    config = kwargs["log_config"]

    pipeline, raw_params = create_pipeline(pipeline_comps)
    pipeline_name = "-".join([name for name, comp in pipeline.steps])

    with mlflow.start_run(experiment_id = config["experiment_id"]) as run:
        mlflow.set_tag("Pipeline Name", pipeline_name)
        start = time()

        # Check if PCA has n_components greater than the number of samples provided for training
        # If so, return infinity as the loss value
        pca = None
        for name, comp in pipeline.steps:
            if name != "PCA":
                continue
            pca = comp

        # Will use number of samples / number of CV splits to estimate sample size during CV
        fold_max_n = len(X_train.groupby(X_train.index)) // kfold.n_splits
        if pca and (pca.n_components >= fold_max_n):
            cv_results_agg = {}
            
            return {
                "loss": np.inf,
                "cv_results": cv_results_agg,
                "time_taken": 0,
                "pipeline": pipeline,
                "status": STATUS_OK,
            }
        
        try:
            start = time()
            cv_results = cross_group_validate(
                estimator = pipeline,
                X = X_train,
                y = y_train,
                cv = kfold,
                groups = X_train.index,
                scoring = scoring,
            )
            time_taken = time() - start

            # Aggregate results across all folds
            cv_results_agg = {
                name: np.mean(value)
                for name, value in cv_results.items()
            }
            
            performance = cv_results_agg[criteria]
            if "neg" in criteria:
                loss = -performance
            elif criteria in ["r2", "max_error", "explained_variance", "d2_absolute_error_score"]:
                loss = performance
            else:
                loss = 1 - performance
            
            pipeline_parameters = pipeline.get_params()
            copied_pipeline = clone(pipeline)
            copied_pipeline.set_params(**pipeline_parameters)

            if config["log_metrics"]:
                mlflow.log_metrics(cv_results_agg)

            if config["log_time"]:
                mlflow.log_metric("time taken", time_taken)

            if config["log_loss"]:
                mlflow.log_metric("loss", loss)

            if config["log_hyperparams"]:
                mlflow.log_params(raw_params)
        
            mlflow.end_run()
        
            return {
                "loss": loss,
                "cv_results": cv_results_agg,
                "time_taken": time_taken,
                "pipeline": copied_pipeline,
                "status": STATUS_OK,
                "exception": None
            }

        except Exception as e:
            cv_results_agg = {}
            return {
                "loss": np.inf,
                "cv_results": cv_results_agg,
                "time_taken": 0,
                "pipeline": pipeline,
                "status": STATUS_OK,
                "exception": e
            }

    return result

    
def stopping_criteria(num_explore: int, num_exploit: int, stagnant_runs: int):
    """
    Defines a custom stopping criteria based on number of runs without improvement,
    but only counted from the exploitation phase onwards
    """
    
    # Needed to allow pickability
    global stop_fn
    def stop_fn(trials, **kwargs):
        all_trials = trials.trials
        best_trial_idx = trials.best_trial["tid"]
        num_since_improvement = len(all_trials) - best_trial_idx

        return (
            (best_trial_idx > num_explore) and 
            (num_since_improvement > stagnant_runs)
        ), kwargs

    return stop_fn


def cross_group_validate(
    estimator: Pipeline,
    X: pd.DataFrame, 
    y: pd.Series,
    cv: BaseCrossValidator,
    groups: pd.Series,
    scoring: list = None,
) -> dict:
    """
    Performs group cross validation on a validation set that is aggregated by
    scanning groups. Returns only the scores defined by the `scoring` parameter.
    The original estimator is left unchanged, so it will have to be refitted

    Inputs
        estimator: The pipeline to be fitted and scored at each fold
        X, y: Training features and labels
        groups: Scanning groups for the data
        scoring: List of tuples in the form (metric_name: str, metric_fn: Callable)
        cv: Predefined cross validation splitter (GroupKFold/StratifiedGroupKFold)
    Outputs
        Dictionary containing the training and validation scores in the format
        {
            "train_metric1": [fold1_metric, fold2_metric, ...],
            "test_metric1": [fold1_metric, fold2_metric, ...],
            ...
        }
    """
    
    if scoring is None: 
        fold_results = {
            f"{set_name}_score": []
            for set_name in ["train", "test"]
        }
    else:
        fold_results = {
            f"{set_name}_{metric_name}": []
            for metric_name in scoring 
            for set_name in ["train", "test"]
        }

    for train_idx, test_idx in cv.split(X, y, groups = groups):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]

        test_groups = groups[test_idx]
        X_test = X.iloc[test_idx, :].groupby(test_groups).mean()
        y_test = y.iloc[test_idx].groupby(test_groups).mean()

        X_unique_index, X_unique_counts = np.unique(X_test.index, return_counts = True)
        y_unique_index, y_unique_counts = np.unique(y_test.index, return_counts = True)

        if not (all(X_unique_counts == 1) and all(y_unique_counts == 1)):
            raise AutoMLException("Validation groups have duplicate groups")

        fold_estimator = clone(estimator).fit(X_train, y_train)
        if scoring is None:
            train_score = fold_estimator.score(X_train, y_train)
            test_score = fold_estimator.score(X_test, y_test)

            fold_results["train_score"].append(train_score)
            fold_results["test_score"].append(test_score)

        else:
            for metric_name in scoring:
                train_pred = fold_estimator.predict(X_train).ravel()
                test_pred = fold_estimator.predict(X_test).ravel()

                if any(np.isnan(train_pred)) or any(np.isnan(test_pred)):
                    fold_results[f"train_{metric_name}"].append(-np.inf)
                    fold_results[f"test_{metric_name}"].append(-np.inf)
                    continue

                scorer = get_scorer(metric_name)
                train_score = scorer._score_func(y_train, train_pred, **scorer._kwargs)
                test_score = scorer._score_func(y_test, test_pred, **scorer._kwargs)

                
                scorer = get_scorer(metric_name)
                train_score = scorer(fold_estimator, X_train, y_train)
                test_score = scorer(fold_estimator, X_test, y_test)

                fold_results[f"train_{metric_name}"].append(train_score)
                fold_results[f"test_{metric_name}"].append(test_score)
                
    return fold_results


def cross_group_predict(
    estimator: Pipeline,
    X: pd.DataFrame, 
    y: pd.Series,
    cv: BaseCrossValidator,
    groups: pd.Series,
) -> pd.Series:
    """
    Performs group cross validation on a validation set that is aggregated by
    scanning groups. Returns the cross validated, GROUPED predictions for each fold.
    Basically mimics sklearn's cross_val_predict with our "train on specimen, 
    test on lot" workflow 

    Inputs
        estimator: The pipeline to be fitted and scored at each fold
        X, y: Training features and labels. Should be specimen level, not lot level
        groups: Scanning groups for the data
        cv: Predefined cross validation splitter (GroupKFold/StratifiedGroupKFold)
    Outputs
        pandas Series containing the cross validated, grouped predictions
    """


    fold_predictions = []
    for train_idx, test_idx in cv.split(X, y, groups = groups):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]

        test_groups = groups[test_idx]
        X_test = X.iloc[test_idx, :].groupby(test_groups).mean()
        y_test = y.iloc[test_idx].groupby(test_groups).mean()

        X_unique_index, X_unique_counts = np.unique(X_test.index, return_counts = True)
        y_unique_index, y_unique_counts = np.unique(y_test.index, return_counts = True)

        if not (all(X_unique_counts == 1) and all(y_unique_counts == 1)):
            raise AutoMLException("Validation groups cannot have duplicate groups")

        fold_estimator = clone(estimator).fit(X_train, y_train)
        test_pred = pd.Series(fold_estimator.predict(X_test), index = X_test.index)
        fold_predictions.append(test_pred)
        
    return pd.concat(fold_predictions).sort_index()

def create_custom_pipeline(sample: dict) -> Pipeline: ## added feature mask
    """
    Creates a pipeline from the search space sampled by hyperopt

    Inputs
        sample: Dictionary containing the sampled parameters from the search space
    Outputs
        Pipeline containing the components and parameters sampled by hyperopt
    """
    pipeline = Pipeline([])
    hyperparams = {}
    for component_type in ["preprocessing", "dim_red"]:
        component_name, component, params = sample[component_type]
 
        if component_type is "preprocessing" and component is None:

            """
            feature mask added here
            """

            # add in feature mask <-- added!!!
            base_wavelengths = np.arange(480, 1051, 3)
                #if component name is tuple
            if isinstance(component_name, tuple):
                window = np.floor(params[1]["window"]/2).astype(int)
                curr_wavelengths = base_wavelengths[window:-window]

            elif component_name =="SG1D" or component_name =="SG2D":
                window = np.floor(params["window"]/2).astype(int)
                curr_wavelengths = base_wavelengths[window:-window]
            else:
                curr_wavelengths = base_wavelengths

            mask = (curr_wavelengths < 750) | (curr_wavelengths > 950)
            pipeline.steps.append(('mask',FeatureMask(mask)))

            """
            end of feature mask
            """
        if component is None:
            continue

        if isinstance(component, tuple):
            for comp_name, comp, param in zip(component_name, component, params):
                unpacked_params = {
                    param_name: param_value.item()
                        if isinstance(param_value, np.ndarray) 
                        else param_value
                    for param_name, param_value in param.items()
                }
                hyperparams.update(unpacked_params)

                comp.set_params(**unpacked_params)
                pipeline.steps.append((comp_name, comp))

        else:
            unpacked_params = {
                param_name: param_value.item()
                    if isinstance(param_value, np.ndarray) 
                    else param_value
                for param_name, param_value in params.items()
            }
            hyperparams.update(unpacked_params)

            component.set_params(**unpacked_params)
            pipeline.steps.append((component_name, component))

        # Add in feature mask and scaler after preprocessing
        if component_type == "preprocessing":

            """
            feature mask added here
            """

            # add in feature mask <-- added!!!
            base_wavelengths = np.arange(480, 1051, 3)
                #if component name is tuple
            if isinstance(component_name, tuple):
                window = np.floor(params[1]["window"]/2).astype(int)
                curr_wavelengths = base_wavelengths[window:-window]

            elif component_name =="SG1D" or component_name =="SG2D":
                window = np.floor(params["window"]/2).astype(int)
                curr_wavelengths = base_wavelengths[window:-window]
            else:
                curr_wavelengths = base_wavelengths

            mask = (curr_wavelengths < 750) | (curr_wavelengths > 950)
            pipeline.steps.append(('mask',FeatureMask(mask)))

            """
            end of feature mask
            """

            pipeline.steps.append(("Scaler", StandardScaler(with_std = False)))

    # print("Pipeline steps",pipeline.named_steps.keys())

    model_name, model, model_params = sample["model"]
    hyperparams.update(model_params)
    model.set_params(**model_params)

    pipeline.steps.append((model_name, model))

    return clone(pipeline), hyperparams


def custom_loss_function(pipeline_comps: dict, **kwargs) -> dict: ## changed to custom pipeline with feature mask and custom loss function
    """
    Defines the loss function used to evaluate the pipeline sampled by hyperopt.
    Loss is defined as cross validated loss on the training data

    Results are returned as a dictionary to be stored in hyperopt's Trial object
    Inputs
        pipeline_comps: The sampled search space
        **kwargs: Additional parameters to be used in evaluation of the pipeline
    Outputs
        Dictionary containing the loss of the current trial and pipeline used in the current
        trial. 
    """
    
    X_train, y_train = kwargs["X_train"], kwargs["y_train"]
    scoring = kwargs["scoring"]
    criteria = kwargs["criteria"]
    kfold = kwargs["kfold"]

    pipeline, raw_params = create_custom_pipeline(pipeline_comps) #changed to custom pipeline

    # Check if PCA has n_components greater than the number of samples provided for training
    # If so, return infinity as the loss value
    pca = None
    for name, comp in pipeline.steps:
        if name != "PCA":
            continue
        pca = comp

    # Number of samples to minimally have is the grouped dataset's amount, 
    # not the sample based one
    fold_max_n = len(X_train.groupby(X_train.index)) // kfold.n_splits

    if pca and (pca.n_components >= fold_max_n):
        cv_results_agg = {}
        
        return {
            "loss": np.inf,
            "cv_results": cv_results_agg,
            "time_taken": 0,
            "pipeline": pipeline,
            "status": STATUS_OK,
            "exception": None
        }

    try:
        start = time()
        cv_results = cross_group_validate(
            estimator = pipeline,
            X = X_train,
            y = y_train,
            cv = kfold,
            groups = X_train.index,
            scoring = scoring,
        )
        time_taken = time() - start

        # Aggregate results across all folds
        cv_results_agg = {
            name: np.mean(value)
            for name, value in cv_results.items()
        }
        
        # performance = cv_results_agg[criteria]
        # if "neg" in criteria:
        #     loss = -performance
        # elif criteria in ["r2", "max_error", "explained_variance", "d2_absolute_error_score"]:
        #     loss = performance
        # else:
        #     loss = 1 - performance

        """
        custom function to calculate maxmisie recall of class 1
        """

        cv_preds = cross_group_predict(
            estimator = pipeline,
            X = X_train,
            y = y_train,
            cv = kfold,
            groups = X_train.index,
        )

        # maximise recall of class 1
        def recall_class(y_true: pd.Series, y_pred: pd.Series, target_class: int) -> float:
            """
            Calculate recall for a given class.
            
            Recall = TP / (TP + FN)
            
            Parameters:
            y_true (pd.Series): True labels (0 or 1)
            y_pred (pd.Series): Predicted labels (0 or 1)
            target_class (int): The class for which recall is calculated
            
            Returns:
            float: Recall score for the specified class
            """
            tp = ((y_true == target_class) & (y_pred == target_class)).sum()  # True Positives
            fn = ((y_true == target_class) & (y_pred != target_class)).sum()  # False Negatives
            
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        recall_class_0 = recall_class(y_train, cv_preds, 0)
        recall_class_1 = recall_class(y_train, cv_preds, 1)

        # weight recall of class 1 higher
        # loss = 1 - 0.95 * recall_class_1 - 0.05 * recall_class_0
        loss = 1 - recall_class_1


        """
        end of custom loss
        """
        
        pipeline_parameters = pipeline.get_params()
        copied_pipeline = clone(pipeline)
        copied_pipeline.set_params(**pipeline_parameters)
    
        return {
            "loss": loss,
            "cv_results": cv_results_agg,
            "time_taken": time_taken,
            "pipeline": copied_pipeline,
            "status": STATUS_OK,
            "exception": None
        }

    except Exception as e:
        cv_results_agg = {}
        return {
            "loss": np.inf,
            "cv_results": cv_results_agg,
            "time_taken": 0,
            "pipeline": pipeline,
            "status": STATUS_OK,
            "exception": e
        }

