import numpy as np

from time import time

from hyperopt import STATUS_OK
from sklearn.base import clone
from func_timeout import func_timeout, FunctionTimedOut

from pptoolbox.platform.automl_v4.utils import create_pipeline, cross_group_validate


def loss_function_timeout(pipeline_comps: dict, **kwargs) -> dict:
    X_train, y_train = kwargs["X_train"], kwargs["y_train"]
    scoring = kwargs["scoring"]
    criteria = kwargs["criteria"]
    kfold = kwargs["kfold"]
    timeout = kwargs["timeout"]

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
        cv_results = func_timeout(
            timeout,
            cross_group_validate,
            kwargs = {
                "estimator": pipeline,
                "X": X_train,
                "y": y_train,
                "cv": kfold,
                "groups": X_train.index,
                "scoring": scoring,
            }
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
            "exception": None,
        }

    except FunctionTimedOut as e:
        cv_results_agg = {}
        return {
            "loss": np.inf,
            "cv_results": cv_results_agg,
            "time_taken": timeout,
            "pipeline": pipeline,
            "status": STATUS_OK,
            "exception": e,
        }

    except Exception as e:
        cv_results_agg = {}
        return {
            "loss": np.inf,
            "cv_results": cv_results_agg,
            "time_taken": 0,
            "pipeline": pipeline,
            "status": STATUS_OK,
            "exception": e,
        }