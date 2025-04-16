import numpy as np
from pptoolbox.utils import WAVELENGTHS_3NM_V1
from pptoolbox.preprocessing import SNV, SavitzkyGolay, FeatureMask
from sklearn.preprocessing import StandardScaler


wavelengths = np.array(WAVELENGTHS_3NM_V1)

FEATURE_MASK_CATEGORIES = {
    0: (wavelengths >= 480) & (wavelengths <= 1050),  # Full spectrum
    1: (wavelengths >= 750) & (wavelengths <= 1000),  # NIR only
    2: (wavelengths >= 480) & (wavelengths <= 750),  # VIS only
}

SPECTRAL_PREPROCESSORS = {
    "SNV": SNV(),
    "SG": SavitzkyGolay(),
    "SNV-SG": [("snv", SNV()), ("sg", SavitzkyGolay())],
}

SPECTRAL_PREPROCESSORS_GRID = {
    "SNV": None,
    "SG": [
        ["suggest_int", {"name": "window", "low": 5, "high": 29, "step": 2}],
        ["suggest_int", {"name": "deriv", "low": 1, "high": 2}],
        ["suggest_int", {"name": "polyorder", "low": 2, "high": 3}],
    ],
}


def construct_preprocessor(trial) -> list:
    pipeline_steps = []
    # Feature mask
    mask_id = trial.suggest_categorical("mask", list(FEATURE_MASK_CATEGORIES.keys()))
    pipeline_steps.append(("mask", FeatureMask(mask=FEATURE_MASK_CATEGORIES[mask_id])))

    # Spectral preprocessor
    preprocessor_name = trial.suggest_categorical(
        "preprocessor", SPECTRAL_PREPROCESSORS.keys()
    )
    preprocessor = SPECTRAL_PREPROCESSORS[preprocessor_name]
    if "SG" in preprocessor_name:
        preprocessor_params = {}
        for method, kwargs in SPECTRAL_PREPROCESSORS_GRID["SG"]:
            preprocessor_params[kwargs["name"]] = getattr(trial, method)(**kwargs)
        if isinstance(preprocessor, list):
            preprocessor[-1][1].set_params(**preprocessor_params)
            pipeline_steps.extend(preprocessor)
        else:
            preprocessor.set_params(**preprocessor_params)
            pipeline_steps.append((preprocessor_name, preprocessor))
    else:
        pipeline_steps.append((preprocessor_name, preprocessor))

    pipeline_steps.append(("scaler", StandardScaler(with_std=False)))
    return pipeline_steps


def construct_preprocessor_from_params(params: dict) -> list:
    pipeline_steps = []
    # Feature mask
    mask_id = params["mask"]
    pipeline_steps.append(("mask", FeatureMask(mask=FEATURE_MASK_CATEGORIES[mask_id])))

    # Spectral preprocessor
    preprocessor_name = params["preprocessor"]
    preprocessor = SPECTRAL_PREPROCESSORS[preprocessor_name]
    if "SG" in preprocessor_name:
        preprocessor_params = {}
        for _, kwargs in SPECTRAL_PREPROCESSORS_GRID["SG"]:
            preprocessor_params[kwargs["name"]] = params[kwargs["name"]]
        if isinstance(preprocessor, list):
            preprocessor[-1][1].set_params(**preprocessor_params)
            pipeline_steps.extend(preprocessor)
        else:
            preprocessor.set_params(**preprocessor_params)
            pipeline_steps.append((preprocessor_name, preprocessor))
    else:
        pipeline_steps.append((preprocessor_name, preprocessor))

    pipeline_steps.append(("scaler", StandardScaler(with_std=False)))
    return pipeline_steps
