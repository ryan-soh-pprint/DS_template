from time import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator, PredefinedSplit
import warnings

def cross_group_validate(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    groups: pd.Series = None,
    scoring: list = None,
) -> dict:
    """
    Performs group cross validation on a validation set that is aggregated by
    scanning groups. Returns only the scores defined by the `scoring` parameter.
    The original estimator is left unchanged, so it will have to be refitted

    Inputs
        estimator: The estimator to be fitted and scored at each fold
        X, y: Training features and labels
        groups: Scanning groups for the data
        scoring: List of tuples in the form (metric_name: str, metric_fn: Callable)
        cv: Predefined cross validation splitter (GroupKFold)
    Outputs
        Dictionary containing the training and validation scores in the format
        {
            "train_score_{metric_name}_split{i}": float,
            "test_score_{metric_name}_split{i}": float,
        }
        pandas Series containing the cross validated, grouped predictions
    """

    if scoring is None:
        fold_results = {f"{set_name}_score_full": [] for set_name in ["train", "test"]}
    else:
        fold_results = {
            f"{set_name}_{metric_name}_full": []
            for metric_name in scoring
            for set_name in ["train", "test", "diff"]
        }

    if isinstance(cv, PredefinedSplit):
        splitter = cv.split(X, y)
    else:
        splitter = cv.split(X, y, groups=groups)

    fit_time = []
    fold_predictions = []

    for i, (train_idx, test_idx) in enumerate(splitter):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]

        test_groups = groups[test_idx]
        X_test = X.iloc[test_idx, :].groupby(test_groups).mean()
        y_test = y.iloc[test_idx].groupby(test_groups).mean()

        _, X_unique_counts = np.unique(X_test.index, return_counts=True)
        _, y_unique_counts = np.unique(y_test.index, return_counts=True)

        assert all(X_unique_counts == 1) and all(
            y_unique_counts == 1
        ), "Validation groups have duplicate groups"

        start = time()
        fold_estimator = clone(estimator).fit(X_train, y_train)

        test_pred = pd.Series(fold_estimator.predict(X_test).ravel(), index = X_test.index)
        fold_predictions.append(test_pred)

        fit_time.append(time() - start)

        if scoring is None:
            train_score = fold_estimator.score(X_train, y_train)
            test_score = fold_estimator.score(X_test, y_test)

            fold_results["train_score_full"].append(train_score)
            fold_results["test_score_full"].append(test_score)
            fold_results[f"train_score_split{i}"] = train_score
            fold_results[f"test_score_{i}"] = test_score

        else:
            train_pred = fold_estimator.predict(X_train).ravel()
            test_pred = fold_estimator.predict(X_test).ravel()
            for metric_name in scoring:
                if any(np.isnan(train_pred)) or any(np.isnan(test_pred)):
                    fold_results[f"train_{metric_name}_full"].append(-np.inf)
                    fold_results[f"test_{metric_name}_full"].append(-np.inf)
                    fold_results[f"diff_{metric_name}_full"].append(np.inf)
                    fold_results[f"train_{metric_name}_split{i}"] = -np.inf
                    fold_results[f"test_{metric_name}_split{i}"] = -np.inf
                    fold_results[f"diff_{metric_name}_split{i}"] = np.inf
                    continue

                scorer = get_scorer(metric_name)
                train_score = scorer._score_func(y_train, train_pred, **scorer._kwargs)
                test_score = scorer._score_func(y_test, test_pred, **scorer._kwargs)

                if "neg" in metric_name:
                    train_score = np.negative(train_score)
                    test_score = np.negative(test_score)

                fold_results[f"train_{metric_name}_full"].append(train_score)
                fold_results[f"test_{metric_name}_full"].append(test_score)
                fold_results[f"diff_{metric_name}_full"].append(
                    abs(train_score - test_score)
                )
                fold_results[f"train_{metric_name}_split{i}"] = train_score
                fold_results[f"test_{metric_name}_split{i}"] = test_score
                fold_results[f"diff_{metric_name}_split{i}"] = abs(
                    train_score - test_score
                )

    if scoring is None:
        train_scores = fold_results.pop("train_score_full")
        test_scores = fold_results.pop("test_score_full")
        diff_scores = fold_results.pop("diff_score_full")
        fold_results["mean_train_score"] = np.mean(train_scores)
        fold_results["std_train_score"] = np.std(train_scores)
        fold_results["mean_test_score"] = np.mean(test_scores)
        fold_results["std_test_score"] = np.std(test_scores)
        fold_results["mean_diff_score"] = np.mean(diff_scores)
        fold_results["std_diff_score"] = np.std(diff_scores)

    else:
        for metric_name in scoring:
            train_scores = fold_results.pop(f"train_{metric_name}_full")
            test_scores = fold_results.pop(f"test_{metric_name}_full")
            diff_scores = fold_results.pop(f"diff_{metric_name}_full")
            fold_results[f"mean_train_{metric_name}"] = np.mean(train_scores)
            fold_results[f"std_train_{metric_name}"] = np.std(train_scores)
            fold_results[f"mean_test_{metric_name}"] = np.mean(test_scores)
            fold_results[f"std_test_{metric_name}"] = np.std(test_scores)
            fold_results[f"mean_diff_{metric_name}"] = np.mean(diff_scores)
            fold_results[f"std_diff_{metric_name}"] = np.std(diff_scores)

    fold_results["mean_fit_time"] = np.mean(fit_time)
    fold_results["std_fit_time"] = np.std(fit_time)

    return fold_results, pd.concat(fold_predictions).sort_index()

def cross_group_predict(
    estimator: BaseEstimator,
    X: pd.DataFrame, 
    y: pd.Series,
    cv: BaseCrossValidator,
    groups: pd.Series,
    # mode: Literal["train", "test"] = "test",
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

        # if not (all(X_unique_counts == 1) and all(y_unique_counts == 1)):
        #     raise AutoMLException("Validation groups cannot have duplicate groups")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_estimator = clone(estimator).fit(X_train, y_train)

            cv_pred = pd.Series(fold_estimator.predict(X_test).ravel(), index = X_test.index)
            fold_predictions.append(cv_pred)

    return pd.concat(fold_predictions).sort_index()
