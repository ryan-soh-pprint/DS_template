from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .constants import RANDOM_STATE
from .hyperparameters import (
    ALPHA_GRID,
    C_GRID,
    CCP_ALPHA_GRID,
    DEGREE_GRID,
    GAMMA_GRID,
    L1_RATIO_GRID,
    LOSS_GRID,
    MAX_DEPTH_GRID,
    MIN_SAMPLES_LEAF_GRID,
    N_ESTIMATORS_GRID,
    N_NEIGHBORS_GRID,
    WEIGHTS_GRID,
)

BASE_MODELS = {
    "DecisionTree": DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "ExtraTrees": ExtraTreesClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "GaussianNB": GaussianNB(),
    "GradientBoost": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "LogisticRegression": LogisticRegression(
        penalty="elasticnet",
        random_state=RANDOM_STATE,
        solver="saga",
        class_weight="balanced",
    ),
    "QDA": QuadraticDiscriminantAnalysis(),
    "RandomForest": RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "SGD": SGDClassifier(
        penalty="elasticnet",
        shuffle=False,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "SVC(RBF)": SVC(
        kernel="rbf",
        random_state=RANDOM_STATE,
        probability=True,
        class_weight="balanced",
    ),
    "SVC(Poly)": SVC(
        kernel="poly",
        random_state=RANDOM_STATE,
        probability=True,
        class_weight="balanced",
    ),
    "SVC(Sigmoid)": SVC(
        kernel="sigmoid",
        random_state=RANDOM_STATE,
        probability=True,
        class_weight="balanced",
    ),
}

MODELS_HYPERPARAMETERS_GRID = {
    "DecisionTree": [MAX_DEPTH_GRID, MIN_SAMPLES_LEAF_GRID, CCP_ALPHA_GRID],
    "ExtraTrees": [
        N_ESTIMATORS_GRID,
        MIN_SAMPLES_LEAF_GRID,
        MAX_DEPTH_GRID,
        CCP_ALPHA_GRID,
    ],
    "GaussianNB": [],
    "GradientBoost": [
        N_ESTIMATORS_GRID,
        MIN_SAMPLES_LEAF_GRID,
        MAX_DEPTH_GRID,
        CCP_ALPHA_GRID,
    ],
    "KNN": [N_NEIGHBORS_GRID, WEIGHTS_GRID],
    "LDA": [],
    "LogisticRegression": [C_GRID, L1_RATIO_GRID],
    "QDA": [],
    "RandomForest": [N_ESTIMATORS_GRID, MAX_DEPTH_GRID, CCP_ALPHA_GRID],
    "SGD": [LOSS_GRID, L1_RATIO_GRID, ALPHA_GRID],
    "SVC(RBF)": [C_GRID, GAMMA_GRID],
    "SVC(Poly)": [C_GRID, GAMMA_GRID, DEGREE_GRID],
    "SVC(Sigmoid)": [C_GRID, GAMMA_GRID],
}

# To enforce dimensionality reduction
DIM_RED_MODELS = [
    "LDA",
    "LogisticRegression",
    "QDA",
]


def construct_classifier_step(trial) -> tuple:
    clf_name = trial.suggest_categorical("model", list(BASE_MODELS.keys()))
    hparam_grid = {}
    for method, kwargs in MODELS_HYPERPARAMETERS_GRID[clf_name]:
        hparam_grid[kwargs["name"]] = getattr(trial, method)(**kwargs)
    clf = BASE_MODELS[clf_name]
    clf.set_params(**hparam_grid)
    return (clf_name, clf)


def construct_classifier_step_from_params(params: dict) -> tuple:
    selected_model = params["model"]
    hparam_grid = {}
    for _, kwargs in MODELS_HYPERPARAMETERS_GRID[selected_model]:
        hparam_grid[kwargs["name"]] = params[kwargs["name"]]
    clf = BASE_MODELS[selected_model]
    clf.set_params(**hparam_grid)
    return (selected_model, clf)
