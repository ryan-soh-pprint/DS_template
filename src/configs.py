import os

# Training configs
DIRECTION = "maximize"
N_TRIALS = 200
N_STARTUP_TRIALS = 100
SEED = 42

# Naming configs
MODULE_NAME = "classify"
MODULE_VERSION = "custom-viv"

RESULTS_DIR = "results/"
DATA_DIR = "data/"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
STORAGE_URL = f"sqlite:///{RESULTS_DIR}/optuna_study.sqlite3"
TRACKING_URI = "http://18.141.241.237:5000/"
