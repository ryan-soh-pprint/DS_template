import numpy as np

from hyperopt import hp
from hyperopt.pyll import scope

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# from sklearn.svm.classes import OneClassSVM
# from sklearn.gaussian_process.gpc import GaussianProcessClassifier
# from sklearn.ensemble.voting_classifier import VotingClassifier
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.linear_model import Perceptron
# from sklearn.mixture import DPGMM
# from sklearn.mixture import GMM 
# from sklearn.mixture import GaussianMixture
# from sklearn.mixture import VBGMM

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier    
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB  
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import *
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from pptoolbox.preprocessing import SNV, SavitzkyGolay, Detrend


MULTICLASS_METRICS = [
    "balanced_accuracy", "f1_micro", "precision_micro", "recall_micro"
]

BINARY_METRICS = [
    "balanced_accuracy", "f1_micro", "precision_micro", "recall_micro"
]

REGRESSION_METRICS = [
    "r2", "neg_root_mean_squared_error", "neg_mean_squared_error", "neg_mean_absolute_error", "max_error"
]

RANDOM_STATE = 42
KFOLD = GroupKFold(n_splits = 3)
STRATIFIED_KFOLD = StratifiedGroupKFold(n_splits = 3)
WL_3 = np.arange(480, 1051, 3)


# Implementation of np.logspace in hyperopt (log base 10)
# @scope.define
# def logspace(x):
#     return 10 ** x
 

CHOICES = {
    "window": [5, 7, 11, 13, 15, 17, 19, 21],
    "kernel": ["linear", "rbf"],
}

# Preprocessing Params
PCA_GRID = hp.uniformint("n_components", 2, 32)
WINDOW = hp.choice("window", CHOICES["window"])

# Tree Params
N_ESTIMATORS = hp.uniformint("n_estimators", 5, 200)
MAX_DEPTH = hp.choice("max_depth", [None, hp.uniformint("max_depth_gt_0", 1, 20)])
CCP_ALPHA = scope.logspace(hp.uniform("ccp_alpha", 0, 10))

# MLP Params
MLP_NODES = hp.uniformint("num_layers", 1, 200)

# Quantile Params
QUANTILE = hp.uniform("quantile", 0, 1)

# Reduced Regularisation params
L1_RATIO = hp.uniform("l1_ratio", 0, 1)
C = scope.logspace(hp.uniform("C", -3, 3))
ALPHA = scope.logspace(hp.uniform("alpha", 0, 5))
LAMBDA = scope.logspace(hp.uniform("lambda", 0, 5))

# SVM Params
NU = hp.uniform("nu", 0.0001, 1)
KERNEL = hp.choice("kernel", CHOICES["kernel"])

# Distance Params
POWER = hp.uniformint("power", 1, 10)

# Nearest Neighbour Params
RADIUS = hp.uniform("radius", 1, 100)
N_NEIGHBORS = hp.uniformint("n_neighbors", 2, 15)

# Stopping Params
STOPPING_RUNS = hp.uniformint("n_iter_no_change", 2, 100)


# Ensemble Classifiers
def create_random_forest():
    rf = RandomForestClassifier(random_state = RANDOM_STATE)
    rf.estimators_ = []

    return rf


def create_extra_trees():
    et = ExtraTreesClassifier(random_state = RANDOM_STATE)
    et.estimators_ = []

    return et


def create_gradient_boost():
    gb = GradientBoostingClassifier(random_state = RANDOM_STATE)
    gb.estimators_ = []

    return gb


def create_ada_boost():
    ab = AdaBoostClassifier(random_state = RANDOM_STATE)
    ab.estimators_ = []

    return ab


COMPONENTS = {
    "preprocessing": [
        ["NoPreproc", None, {}],
        # ["Detrend", Detrend(), {}],
        # ["SNV", SNV(), {}],
        # ["SG1D", SavitzkyGolay(deriv = 1), {"window": WINDOW}],
        # ["SG2D", SavitzkyGolay(deriv = 2), {"window": WINDOW}],
        # [["SNV", "SG1D"], [SNV(), SavitzkyGolay(deriv = 1)], [{}, {"window": WINDOW}]],
        # [["SNV", "SG2D"], [SNV(), SavitzkyGolay(deriv = 2)], [{}, {"window": WINDOW}]],
    ],
    "dimred_choice": [
        ["PCA", PCA(random_state = RANDOM_STATE), {"n_components": PCA_GRID}],
        ["NoPCA", None, {}],
    ],
}

# All models use all preprocessing methods
preprocessing = hp.choice("preprocessing", COMPONENTS["preprocessing"])

dimred_choice = hp.choice("dimred_choice", COMPONENTS["dimred_choice"])
dimred_yes = ["PCA", PCA(random_state = RANDOM_STATE), {"n_components": PCA_GRID}]
dimred_no = ["NoPCA", None, {}]


CLASSIFIERS = {
    "LogisticRegression": [
        "LogisticRegression",
        LogisticRegression(penalty = "elasticnet", random_state = RANDOM_STATE, solver = "saga"),
        {"C": C, "l1_ratio": L1_RATIO},
    ],
    "SGD": [
        "SGD",
        SGDClassifier(penalty = "elasticnet", shuffle = False, loss = "modified_huber"),
        {"l1_ratio": L1_RATIO, "alpha": ALPHA},
    ],
    "KNN": [
        "KNN",
        KNeighborsClassifier(),
        {"n_neighbors": N_NEIGHBORS, "p": POWER},
    ],
    "RadiusNeighbors": [
        "RadiusNeighbors",
        RadiusNeighborsClassifier(),
        {"radius": RADIUS},
    ],
    "SVC": [
        "SVC", 
        SVC(random_state = RANDOM_STATE, probability = True, cache_size = 2000),
        {"C": C, "kernel": KERNEL}    
    ],
    "NuSVC": [
        "NuSVC",
        NuSVC(random_state = RANDOM_STATE, probability = True),
        {"nu": NU, "kernel": KERNEL},
    ],
    "GaussianNB": [
        "GaussianNB",
        GaussianNB(),
        {},
    ],
    "MultinomialNB": [ 
        "MultinomialNB",
        MultinomialNB(force_alpha = True),
        {"alpha": ALPHA},
    ],
    "DecisionTree": [
        "DecisionTree", 
        DecisionTreeClassifier(random_state = RANDOM_STATE), 
        {"max_depth": MAX_DEPTH, "ccp_alpha": CCP_ALPHA},
    ],
    "ExtraTree": [
        "ExtraTree", 
        ExtraTreeClassifier(random_state = RANDOM_STATE), 
        {"max_depth": MAX_DEPTH, "ccp_alpha": CCP_ALPHA},
    ],
    "RandomForest": [
        "RandomForest",
        create_random_forest(),
        {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH, "ccp_alpha": CCP_ALPHA},
    ],
    "ExtraTrees": [
        "ExtraTrees",
        create_extra_trees(),
        {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH, "ccp_alpha": CCP_ALPHA},
    ],
    "GradientBoost": [
        "GradientBoost",
        create_gradient_boost(),
        {"n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH, "ccp_alpha": CCP_ALPHA},
    ],
    "AdaBoost": [
        "AdaBoost",
        create_ada_boost(),
        {"n_estimators": N_ESTIMATORS},
    ],
    "LDA": [
        "LDA", 
        LinearDiscriminantAnalysis(), 
        {},
    ],
    "QDA": [
        "QDA", 
        QuadraticDiscriminantAnalysis(), 
        {}
    ],
    "MLP": [
        "MLP", 
        MLPClassifier(early_stopping = True, random_state = RANDOM_STATE, shuffle = False), 
        {"hidden_layer_sizes": (MLP_NODES, ), "n_iter_no_change": STOPPING_RUNS, "alpha": ALPHA},
    ],
}

REGRESSORS = {
    "LinearRegression": [
        "LinearRegression",
        LinearRegression(),
        {},
    ],
    "Enet": [
        "Enet", 
        ElasticNet(), 
        {"l1_ratio": L1_RATIO, "alpha": ALPHA}
    ],
    "Lars": [
        "Lars",
        Lars(random_state = RANDOM_STATE),
        {}
    ],
    "LassoLars": [
        "LassoLars",
        LassoLars(random_state = RANDOM_STATE),
        {"alpha": ALPHA}
    ],
    "OMP": [
        "OMP",
        OrthogonalMatchingPursuit(),
        {}
    ],
    "BayesianRidge": [
        "BayesianRidge",
        BayesianRidge(),
        {"alpha_1": ALPHA, "alpha_2": ALPHA, "lambda_1": LAMBDA, "lambda_2": LAMBDA}
    ],
    "GLM": [
        "GLM",
        GammaRegressor(),
        {"alpha": ALPHA}
    ],
    "SGD": [
        "SGD",
        SGDRegressor(penalty = "elasticnet", shuffle = False, early_stopping = True, random_state = RANDOM_STATE),
        {"alpha": ALPHA, "l1_ratio": L1_RATIO, "n_iter_no_change": STOPPING_RUNS}
    ],
    "Paggro": [
        "Paggro",
        PassiveAggressiveRegressor(early_stopping = True, shuffle = False, random_state = RANDOM_STATE),
        {"C": C, "n_iter_no_change": STOPPING_RUNS}
    ],
    "Quantile": [
        "Quantile",
        QuantileRegressor(solver = "highs"),
        {"alpha": ALPHA, "quantile": QUANTILE},
    ],
    "KNN": [
        "KNN",
        KNeighborsRegressor(),
        {"n_neighbors": N_NEIGHBORS, "p": POWER},
    ],
    "RadiusNeighbors": [
        "RadiusNeighbors",
        RadiusNeighborsRegressor(),
        {"radius": RADIUS, "p": POWER},
    ],
    "PLS": [
        "PLS",
        PLSRegression(scale = False),
        {"n_components": PCA_GRID},
    ],
    "SVR": [
        "SVR", 
        SVR(cache_size = 2000),
        {"C": C, "kernel": KERNEL},
    ],
    "NuSVR": [
        "NuSVR", 
        NuSVR(), 
        {"C": C, "kernel": KERNEL},
    ],
    "MLP": [
        "MLP",
        MLPRegressor(early_stopping = True, random_state = RANDOM_STATE, shuffle = False),
        {"hidden_layer_sizes": (MLP_NODES, ), "n_iter_no_change": STOPPING_RUNS, "alpha": ALPHA}
    ],
}

### Hyperopt samples with lists instead of dictionaries.
CLASSIFIER_PIPELINES = hp.choice("classifier", [
    {
        "model": CLASSIFIERS["LogisticRegression"],
        "dim_red": dimred_yes,
        "preprocessing": preprocessing,
    },
    {
        "model": CLASSIFIERS["SGD"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # {
    #     "model": CLASSIFIERS["Paggro"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # {
    #     "model": CLASSIFIERS["NearestCentroid"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # Removed Nearest Neighbors to ensure low datasets do not raise this issue
    # {
    #     "model": CLASSIFIERS["KNN"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # {
    #     "model": CLASSIFIERS["RadiusNeighbors"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": CLASSIFIERS["SVC"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # NuSVC's `nu` parameter depends too heavily on the dataset
    # NuSVC is basically a constrained version of SVC that does not allow for more
    # than `nu` proportion of misclassification
    # {
    #     "model": CLASSIFIERS["NuSVC"]
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": CLASSIFIERS["GaussianNB"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # MultinomialNB does not allow for negative values in the dataset
    # {
    #     "model": CLASSIFIERS["MultinomialNB"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": CLASSIFIERS["DecisionTree"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # {
    #     "model": CLASSIFIERS["ExtraTree"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # No ensemble methods
    # {
    #     "model": CLASSIFIERS["RandomForest"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # {
    #     "model": CLASSIFIERS["ExtraTrees"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # {
    #     "model": CLASSIFIERS["GradientBoost"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # {
    #     "model": CLASSIFIERS["AdaBoost"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": CLASSIFIERS["LDA"],
        "dim_red": dimred_yes,
        "preprocessing": preprocessing,
    },
    {
        "model": CLASSIFIERS["QDA"],
        "dim_red": dimred_yes,
        "preprocessing": preprocessing,
    },
    # No MLP Models
    # {
    #     "model": CLASSIFIERS["MLP"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
])

REGRESSION_PIPELINES = hp.choice("regressor", [
    # Remove OLS
    # {
    #     "model": REGRESSORS["LinearRegression"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": REGRESSORS["Enet"],
        "dim_red": dimred_yes,
        "preprocessing": preprocessing,
    },
    {
        "model": REGRESSORS["Lars"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # Poor performance 
    # {
    #     "model": REGRESSORS["LassoLars"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": REGRESSORS["OMP"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    {
        "model": REGRESSORS["BayesianRidge"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # Some y values are out of valid range of HalfGammaLoss
    # {
    #     "model": REGRESSORS["GLM"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": REGRESSORS["SGD"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    {
        "model": REGRESSORS["Paggro"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # Poor performance
    # {
    #     "model": REGRESSORS["Quantile"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # Scared overfit
    # {
    #     "model": REGRESSORS["KNN"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # Scared overfit
    # {
    #     "model": REGRESSORS["RadiusNeighbors"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    {
        "model": REGRESSORS["PLS"],
        "dim_red": dimred_no,
        "preprocessing": preprocessing,
    },
    {
        "model": REGRESSORS["SVR"],
        "dim_red": dimred_choice,
        "preprocessing": preprocessing,
    },
    # NuSVC's `nu` parameter depends too heavily on the dataset
    # NuSVC is basically a constrained version of SVC that does not allow for more
    # than `nu` proportion of misclassification
    # {
    #     "model": REGRESSORS["NuSVR"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # },
    # No MLP Models
    # {
    #     "model": REGRESSORS["MLP"],
    #     "dim_red": dimred_choice,
    #     "preprocessing": preprocessing,
    # }
])
