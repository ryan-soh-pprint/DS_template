from sklearn.decomposition import PCA
from umap import UMAP

DIM_RED_MODELS = {
    "pca": PCA(),
    "umap": UMAP(),
}

DIM_RED_HYPERPARAMETERS_GRID = {
    "pca": [
        [
            "suggest_int",
            {"name": "dim_red__n_components", "low": 2, "high": 24, "step": 1},
        ]
    ],
    "umap": [
        [
            "suggest_int",
            {"name": "dim_red__n_components", "low": 2, "high": 24, "step": 1},
        ],
        [
            "suggest_int",
            {"name": "dim_red__n_neighbors", "low": 15, "high": 40, "step": 1},
        ],
    ],
}


def construct_dim_red_step(trial) -> tuple:
    method_name = trial.suggest_categorical(
        "dim_red_method", list(DIM_RED_MODELS.keys())
    )
    dim_red = DIM_RED_MODELS[method_name]
    hyperparameters = {}
    for method, kwargs in DIM_RED_HYPERPARAMETERS_GRID[method_name]:
        hyperparameters[kwargs["name"].split("__")[-1]] = getattr(trial, method)(
            **kwargs
        )
    dim_red.set_params(**hyperparameters)
    return (method_name, dim_red)


def construct_dim_red_step_from_params(params: dict) -> tuple:
    method_name = params["dim_red_method"]
    dim_red = DIM_RED_MODELS[method_name]
    hyperparameters = {}
    for _, kwargs in DIM_RED_HYPERPARAMETERS_GRID[method_name]:
        hyperparameters[kwargs["name"].split("__")[-1]] = params[kwargs["name"]]
    dim_red.set_params(**hyperparameters)
    return (method_name, dim_red)
