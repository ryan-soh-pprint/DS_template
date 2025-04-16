from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek

from .constants import RANDOM_STATE

SAMPLE_WEIGHT_MODELS = ["GaussianNB", "GradientBoost"]

SAMPLER_METHODS = [None, "Under", "Over", "Combined"]

SAMPLERS_BASE = {
    "Under": {
        "ClusterCentroids": ClusterCentroids(random_state=RANDOM_STATE),
        "RandomUnderSampler": RandomUnderSampler(random_state=RANDOM_STATE),
        "TomekLinks": TomekLinks(),
    },
    "Over": {
        "RandomOverSampler": RandomOverSampler(random_state=RANDOM_STATE),
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "ADASYN": ADASYN(random_state=RANDOM_STATE),
    },
    "Combined": {
        "SMOTEENN": SMOTEENN(random_state=RANDOM_STATE),
        "SMOTETomek": SMOTETomek(random_state=RANDOM_STATE),
    },
}


def construct_imbal_sampler_step(trial) -> tuple:
    sampler_method = trial.suggest_categorical("sampler_method", SAMPLER_METHODS)
    if sampler_method is not None:
        samplers_dict = SAMPLERS_BASE[sampler_method]
        sampler_name = trial.suggest_categorical(
            f"sampler_{sampler_method}", list(samplers_dict.keys())
        )
        return (sampler_name, samplers_dict[sampler_name])
    return None


def construct_imbal_sampler_step_from_params(params: dict) -> tuple:
    sampler_method = params["sampler_method"]
    if sampler_method is not None:
        samplers_dict = SAMPLERS_BASE[sampler_method]
        sampler_name = params[f"sampler_{sampler_method}"]
        return (sampler_name, samplers_dict[sampler_name])
    return None
