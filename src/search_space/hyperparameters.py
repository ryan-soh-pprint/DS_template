"""Hyperparameter search space."""

from sklearn.gaussian_process.kernels import RBF, Matern

LENGTH_SCALE_GRID = ["suggest_float", {"name": "length_scale", "low": 1e-3, "high": 1}]
NU_GRID = ["suggest_float", {"name": "nu", "low": 0.01, "high": 0.8}]

KERNEL_CHOICES = ["linear", "poly", "rbf", "sigmoid"]
# KERNEL_INSTANCES = [
#     ["rbf", RBF(), {"length_scale": LENGTH_SCALE_GRID}],
#     ["matern", Matern(), {"length_scale": LENGTH_SCALE_GRID, "nu": NU_GRID}],
# ]
LOSS_CHOICES = ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]

# This is arranged alphabetically
ALPHA_GRID = ["suggest_float", {"name": "alpha", "low": 1e-3, "high": 1e4, "log": True}]
C_GRID = ["suggest_float", {"name": "C", "low": 1e-3, "high": 1e3, "log": True}]
CCP_ALPHA_GRID = [
    "suggest_float",
    {"name": "ccp_alpha", "low": 1e-1, "high": 1e5, "log": True},
]
DEGREE_GRID = ["suggest_int", {"name": "degree", "low": 2, "high": 5}]
GAMMA_GRID = ["suggest_float", {"name": "gamma", "low": 1e-3, "high": 1e2, "log": True}]
# GP_KERNEL_GRID = [
#     "suggest_categorical",
#     {"name": "kernel", "choices": KERNEL_INSTANCES},
# ]
KERNEL_GRID = ["suggest_categorical", {"name": "kernel", "choices": KERNEL_CHOICES}]
L1_RATIO_GRID = ["suggest_float", {"name": "l1_ratio", "low": 0, "high": 1}]
LAMBDA_GRID = [
    "suggest_float",
    {"name": "lambda", "low": 1e-6, "high": 1e6, "log": True},
]
LOSS_GRID = ["suggest_categorical", {"name": "loss", "choices": LOSS_CHOICES}]
MAX_DEPTH_GRID = ["suggest_int", {"name": "max_depth", "low": 1, "high": 12}]
MIN_SAMPLES_LEAF_GRID = [
    "suggest_float",
    {"name": "min_samples_leaf", "low": 0.05, "high": 0.3},
]
N_COMPONENTS_GRID = [
    "suggest_int",
    {"name": "n_components", "low": 2, "high": 20},
]  # 32
N_ESTIMATORS_GRID = ["suggest_int", {"name": "n_estimators", "low": 5, "high": 200}]
N_NEIGHBORS_GRID = ["suggest_int", {"name": "n_neighbors", "low": 4, "high": 20}]  # 32
RADIUS_GRID = ["suggest_float", {"name": "radius", "low": 1e-2, "high": 100}]
WEIGHTS_GRID = [
    "suggest_categorical",
    {"name": "weights", "choices": ["uniform", "distance"]},
]
