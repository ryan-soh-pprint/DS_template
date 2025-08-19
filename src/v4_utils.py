from typing import Literal
from sklearn.model_selection import BaseCrossValidator

import pandas as pd
import numpy as np
import mlflow
import warnings

from math import ceil
from time import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import *
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from hyperopt import STATUS_OK, STATUS_FAIL

from .exceptions import AutoMLException
from .validations import validate_pca
from .constants import EXCLUDED_PIPELINES

import inspect
import itertools

def generate_synthetic_samples(
        y, 
        X,
        base_group_size=2, 
        max_synthetic=7):
    """
    Generate synthetic samples by computing means of groups of samples with the same index.
    
    Parameters:
    -----------
    y : pd.DataFrame
        Target variable data
    X : pd.DataFrame
        Feature data
    base_group_size : int
        Number of samples to combine for each synthetic sample (default: 2)

    Returns:
    --------
    tuple : (y_synthetic, X_synthetic)
        Combined original and synthetic data
    """

    duplicated_y_list = []
    duplicated_X_list = []

    unique_indices = y.index.unique()

    for unique_idx in unique_indices:
        same_idx_data_y = y.loc[unique_idx]
        same_idx_data = X.loc[unique_idx]

        if same_idx_data_y.shape[0] >= base_group_size:
            sample_positions = np.arange(same_idx_data_y.shape[0])
            combinations = np.array(list(itertools.combinations(sample_positions, base_group_size)))

            if max_synthetic is not None and len(combinations) > max_synthetic:
                combinations = combinations[:max_synthetic]

            # Vectorised mean for y
            y_selected = same_idx_data_y.to_numpy()[combinations]
            y_means = y_selected.mean(axis=1)
            if y_means.ndim == 1:  # ensure always 2D
                y_means = y_means[:, None]
            y_synthetic = pd.DataFrame(
                y_means, 
                index=[f"{unique_idx}{base_group_size}"] * len(y_means), 
            )
            duplicated_y_list.append(y_synthetic)

            # Vectorised mean for X
            X_selected = same_idx_data.to_numpy()[combinations]
            X_means = X_selected.mean(axis=1)
            if X_means.ndim == 1:  # ensure always 2D
                X_means = X_means[:, None]
            X_synthetic = pd.DataFrame(
                X_means, 
                index=[f"{unique_idx}{base_group_size}"] * len(X_means), 
                columns=same_idx_data.columns
            )
            duplicated_X_list.append(X_synthetic)

    if duplicated_y_list:
        y_combined = pd.concat([y] + duplicated_y_list)
        X_combined = pd.concat([X] + duplicated_X_list)
    else:
        y_combined = y
        X_combined = X

    y_combined.index = y_combined.index.astype("int64")
    X_combined.index = X_combined.index.astype("int64")

    print(f"Final train set shape: {y_combined.shape}, {sum(len(df) for df in duplicated_y_list)} synthetic samples, {len(y.index.unique())} original unique indices, {len(y_combined.index.unique())} new unique indices")

    return y_combined, X_combined

def fit_pipeline_withsyn(pipeline, X_train, y_train, sample_weights=None):
    """Fit a pipeline with synthetic samples."""
    # Generate synthetic samples
    y_combined, X_combined = generate_synthetic_samples(y_train, X_train)

    # Clone the pipeline first
    pipeline_clone = clone(pipeline)

    fitted_pipeline = pipeline_clone.fit(X_combined, y_combined)
    return fitted_pipeline

def fit_pipeline_safe(pipeline, X_train, y_train, sample_weights = None):
    """Safely fit a pipeline, checking if the last step supports sample weights."""

    last_step_name, last_step = list(pipeline.named_steps.items())[-1]
    
    # Clone the pipeline first
    pipeline_clone = clone(pipeline)
    
    # Check if 'sample_weight' is in the fit signature of the last step
    try:
        if hasattr(last_step, 'fit') and 'sample_weight' in inspect.signature(last_step.fit).parameters:

            ranks = np.argsort(np.argsort(y_train))       # 0 = smallest, n-1 = largest
            q = (ranks + 0.5) / len(y_train)             # normalize to 0–1
            tailness = np.abs(2*q - 1)                   # 0=center, 1=extremes

            alpha = 2.5                                  # controls strength of weighting
            sample_weights = (tailness + 1e-6) ** alpha
            sample_weights /= sample_weights.mean()                     # normalize to mean=1

            kwargs = {f"{last_step_name}__sample_weight": sample_weights}
            print(f"Fitting {last_step_name} with sample weights")
        else:
            kwargs = {}
            print(f"Fitting {last_step_name} without sample weights")
        
        fitted_pipeline = pipeline_clone.fit(X_train, y_train, **kwargs)
        return fitted_pipeline
        
    except Exception as e:
        print(f"Error fitting pipeline: {e}")
        # Fallback: try fitting without sample weightsshoul
        try:
            return pipeline_clone.fit(X_train, y_train)
        except Exception as e2:
            print(f"Error fitting pipeline without sample weights: {e2}")
            raise e2
        
def create_pipeline(sample: dict) -> Pipeline:
    """Creates a pipeline from the search space sampled by hyperopt.
    If the pipeline is invalid, return None

    Inputs
        sample: Dictionary containing the sampled parameters from the search space
    Outputs
        Pipeline containing the components and parameters sampled by hyperopt
    """

    pipeline = Pipeline([])
    # Check if this pipeline is invalid
    for pipeline_dict in EXCLUDED_PIPELINES:
        invalid_pipeline_combi = []
        for key, value in pipeline_dict.items():
            if value == None:
                continue

            invalid_pipeline_combi.append(sample[key][0] == value)

        if all(invalid_pipeline_combi):
            raise AutoMLException("Invalid sampled pipeline")

    for component_type in ["preprocessing", "dim_red", "model"]:
        component_name, component, params = sample[component_type]
        if component is None:
            continue

        unpacked_params = {
            key.split("__")[-1]: value
            for key, value in params.items()
        }
        component.set_params(**unpacked_params)

        if ("SGD" in component_name) and (component_type == "preprocessing"):
            num_deriv = component.deriv
            num_window = component.window
            new_name = f"SG{num_deriv}D{num_window}W"

            component_name = component_name.replace("SGD", new_name)

        pipeline.steps.append((component_name, component))

        # Add in scaler after preprocessing
        if component_type == "preprocessing":
            pipeline.steps.append(("Scaler", StandardScaler(with_std = False)))
    
    return pipeline


def cross_group_predict(
    estimator: Pipeline,
    X: pd.DataFrame, 
    y: pd.Series,
    cv: BaseCrossValidator,
    groups: pd.Series,
    mode: Literal["train", "test"] = "test",
) -> pd.Series:
    """Performs group cross validation on a validation set that is aggregated by
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
    X = X.set_index(groups)
    for train_idx, test_idx in cv.split(X, y, groups = groups):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        test_groups = groups[test_idx]
        X_test = X.iloc[test_idx, :].groupby(test_groups).mean()
        y_test = y.iloc[test_idx].groupby(test_groups).mean()

        X_unique_index, X_unique_counts = np.unique(X_test.index, return_counts = True)
        y_unique_index, y_unique_counts = np.unique(y_test.index, return_counts = True)

        if not (all(X_unique_counts == 1) and all(y_unique_counts == 1)):
            raise AutoMLException("Validation groups cannot have duplicate groups")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # fold_estimator = clone(estimator).fit(X_train, y_train)
            fold_estimator = fit_pipeline_safe(estimator, X_train, y_train)

            cv_pred = pd.Series(fold_estimator.predict(X_test).ravel(), index = X_test.index)
            fold_predictions.append(cv_pred)

    return pd.concat(fold_predictions).sort_index()


def cross_group_validate(
    estimator: Pipeline,
    X: pd.DataFrame, 
    y: pd.Series,
    cv: BaseCrossValidator,
    groups: pd.Series,
    scoring: list = None,
) -> dict:
    """Performs group cross validation on a validation set that is aggregated by
    scanning groups. Returns only the scores defined by the `scoring` parameter.
    The original estimator is left unchanged, so it will have to be refitted

    Inputs
        estimator: The pipeline to be fitted and scored at each fold
        X, y: Training features and labels
        groups: Scanning groups for the data
        scoring: List of sklearn metric names, or tuples in the form 
            (metric_name: str, scorer: Callable). 
            E.g. ["balanced_accuracy", (cohen_kappa, cohen_kappa_score)]
            Function signature for `scorer` must minimally be scorer(estimator, X, y)
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
        fold_results = {}
        for metric_name in scoring:
            for set_name in ["train", "test"]:
                if isinstance(metric_name, tuple):
                    metric_name, scorer = metric_name
                fold_results[f"{set_name}_{metric_name}"] = []

    for train_idx, test_idx in cv.split(X, y, groups = groups):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]

        test_groups = groups[test_idx]
        X_test = X.iloc[test_idx, :].groupby(test_groups).mean()
        y_test = y.iloc[test_idx].groupby(test_groups).mean()

        X_unique_index, X_unique_counts = np.unique(X_test.index, return_counts = True)
        y_unique_index, y_unique_counts = np.unique(y_test.index, return_counts = True)

        if not (all(X_unique_counts == 1) and all(y_unique_counts == 1)):
            raise AutoMLException("Validation groups have duplicate groups")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # fold_estimator = clone(estimator).fit(X_train, y_train)
            fold_estimator = fit_pipeline_safe(estimator, X_train, y_train)
        
            if scoring is None:
                train_score = fold_estimator.score(X_train, y_train)
                test_score = fold_estimator.score(X_test, y_test)

                fold_results["train_score"].append(train_score)
                fold_results["test_score"].append(train_score)

            else:
                train_pred = fold_estimator.predict(X_train).ravel()
                test_pred = fold_estimator.predict(X_test).ravel()

                for metric_name in scoring:
                    if isinstance(metric_name, tuple):
                        metric_name, scorer = metric_name
                    
                    else:
                        scorer = get_scorer(metric_name)

                    if any(np.isnan(train_pred)) or any(np.isnan(test_pred)):
                        fold_results[f"train_{metric_name}"].append(-np.inf)
                        fold_results[f"test_{metric_name}"].append(-np.inf)
                        continue

                    train_score = scorer(fold_estimator, X_train, y_train)
                    test_score = scorer(fold_estimator, X_test, y_test)

                    fold_results[f"train_{metric_name}"].append(train_score)
                    fold_results[f"test_{metric_name}"].append(test_score)


    return fold_results