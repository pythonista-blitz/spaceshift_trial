# import
import datetime
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
import xgboost
from matplotlib import pyplot as plt
from prettytable import RANDOM
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.simplefilter("ignore", category=DeprecationWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# path
ROOT_DIR = Path.cwd().parent.resolve()
print("root path:", ROOT_DIR)
MLRUN_PATH = ROOT_DIR.parents[0] / "mlruns"
if MLRUN_PATH.exists():
    print("MLRUN path:", MLRUN_PATH)
else:
    print("MLRUN path does not exist.")
    exit()

# competition name(= experiment name)
EXPERIMENT_NAME = ROOT_DIR.name
print("experiment name:", EXPERIMENT_NAME)

# print(f"\n GPU SETUP \n")
# if torch.cuda.is_available():
#     print(f"RUNNING ON GPU - {torch.cuda.get_device_name()}")
# else:
#     print(f"RUNNING ON CPU")


# mlflow settings
nowstr = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
mlflow.set_tracking_uri(str(MLRUN_PATH) + "/")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.start_run(run_name=nowstr)


# random seed
RANDOM_SEED = 126
mlflow.log_param(key="random_seed", value=RANDOM_SEED)


mlflow.end_run()
