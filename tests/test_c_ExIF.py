import os
import timeit
import sys
import os
import logging

import numpy as np
import pandas as pd
from tqdm import trange
from scipy.io import loadmat
from glob import glob
from sklearn.preprocessing import StandardScaler


def append_dirname(dirname: str, max_levels: int = 10):
    """Append a directory to the system path."""
    # from the current path, go up to max_levels directories to find the directory to append
    path = os.getcwd()
    for _ in range(max_levels):
        path = os.path.dirname(path)
        logging.debug(os.path.basename(path))
        if os.path.basename(path) == dirname and os.path.isdir(path):
            sys.path.append(str(path))
            return
    raise RuntimeError(f"Could not find directory {dirname} in {max_levels} levels")


append_dirname("ExIFFI")
from utils.utils import partition_data


def get_df_paths():
    path = os.getcwd()
    path = os.path.dirname(path)
    path_real = os.path.join(path, "data", "real")
    mat_files_real = glob(os.path.join(path_real, "*.mat"))
    mat_file_names_real = {os.path.basename(x).split(".")[0]: x for x in mat_files_real}
    csv_files_real = glob(os.path.join(path_real, "*.csv"))
    csv_file_names_real = {os.path.basename(x).split(".")[0]: x for x in csv_files_real}
    # dataset_names = list(mat_file_names_real.keys()) + list(csv_file_names_real.keys())
    mat_file_names_real.update(csv_file_names_real)
    dataset_paths = mat_file_names_real.copy()
    return dataset_paths


def drop_duplicates(X, y):
    S = np.c_[X, y]
    S = pd.DataFrame(S).drop_duplicates().to_numpy()
    X, y = S[:, :-1], S[:, -1]
    return X, y


def load_data(path):
    data = loadmat(path)
    X, y = data["X"], data["y"]
    y = np.hstack(y)
    X, y = drop_duplicates(X, y)
    return X, y


def load_data_csv(path):
    data = pd.read_csv(path, index_col=0)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    X = data[data.columns[data.columns != "Target"]]
    y = data["Target"]

    X, y = drop_duplicates(X, y)

    return X, y


def pre_process(path):
    extension = os.path.splitext(path)[1]

    if extension == ".csv":
        X, y = load_data_csv(path)
    elif extension == ".mat":
        X, y = load_data(path)
    else:
        raise ValueError("Extension not supported")

    X_train, X_test = partition_data(X, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test = np.r_[X_train, X_test]

    return X_train.astype(np.float64), X_test.astype(np.float64)


from Extended_IF import ExtendedIF, ExtendedIF_c

dataset_paths = get_df_paths()
name = "moodify"
savedir = "/dev/null"
n_trees = 100
X_train, X_test = pre_process(dataset_paths[name])
seed = 42


exif = ExtendedIF(
    n_trees=n_trees, max_depth=100, subsample_size=256, plus=1, disable_fit_tqdm=False
)
exif_c = ExtendedIF_c(
    n_trees=n_trees,
    max_depth=100,
    subsample_size=256,
    plus=1,
    num_processes_anomaly=2,
    disable_fit_tqdm=False,
)

# --- Compare the Python and C functions output ---
print("... Model fit in progress ...")
print("X_train shape", X_train.shape)
print("X_test shape", X_test.shape)

print("Start fit normal model")
np.random.seed(seed)
exif.fit(X_train)

print("Start fit c model")
np.random.seed(seed)
exif_c.fit(X_train)


# print("...Start timing tests...")
# num_runs = 100
# c_time = timeit.timeit(
#     lambda: exif_c.Anomaly_Score(X_test),
#     number=num_runs,
# )
# pure_c_time = timeit.timeit(
#     lambda: exif_c.c_AnomalyScore(X_test),
#     number=num_runs,
# )
# print("num runs", num_runs)
# print(f"C time: {c_time/num_runs}")
# print(f"Pure C time: {pure_c_time/num_runs}")
# quit()

print("... Equality check (CTRL+C to skip) ...")
try:
    score_original = exif.Anomaly_Score(X_test)
    score_c = exif_c.Anomaly_Score(X_test)
    score_pure_c = exif_c.c_AnomalyScore(X_test)
    print(
        "Difference in anomaly scores original - c: ",
        np.array(score_original - score_c).mean(),
    )
    print(
        "Difference in anomaly scores original - pure c: ",
        np.array(score_original - score_pure_c).mean(),
    )
except KeyboardInterrupt:
    pass

print("\n... Timing tests ...")
num_runs = 1
python_time = timeit.timeit(
    lambda: exif.Anomaly_Score(X_test),
    number=num_runs,
)

c_time = timeit.timeit(
    lambda: exif_c.Anomaly_Score(X_test),
    number=num_runs,
)

pure_c_time = timeit.timeit(
    lambda: exif_c.c_AnomalyScore(X_test),
    number=num_runs,
)

print("num runs", num_runs)
print(f"Python time: {python_time/num_runs}")
print(f"C time: {c_time/num_runs}")
print(f"Pure C time: {pure_c_time/num_runs}")

print("C speedup:", 100 * python_time / c_time, r"%")
print("Pure C speedup:", 100 * python_time / pure_c_time, r"%")


quit()
# ----------------------------------------------------------
from utils.feature_selection import *
from models import *
from models.Extended_IF import *
from models.Extended_DIFFI_parallel import *
from models.Extended_DIFFI_original import *


def compute_imps(model, X_train, X_test, n_runs):

    imps = np.zeros(shape=(n_runs, X_train.shape[1]))
    for i in trange(n_runs, desc="Fit & Importances"):
        print("Start fit")
        model.fit(X_train)
        print("End fit")
        print("Start Global Importance")
        imps[i, :] = model.Global_importance(
            X_test, calculate=True, overwrite=False, depth_based=False
        )
        print("End Global Importance")

    return imps


if __name__ == "__main__":

    path = os.getcwd()
    path = os.path.dirname(path)
    path_real = os.path.join(path, "data", "real")
    mat_files_real = glob(os.path.join(path_real, "*.mat"))
    mat_file_names_real = {os.path.basename(x).split(".")[0]: x for x in mat_files_real}
    csv_files_real = glob(os.path.join(path_real, "*.csv"))
    csv_file_names_real = {os.path.basename(x).split(".")[0]: x for x in csv_files_real}
    dataset_names = list(mat_file_names_real.keys()) + list(csv_file_names_real.keys())
    mat_file_names_real.update(csv_file_names_real)
    dataset_paths = mat_file_names_real.copy()

    name = "wine"
    savedir = "/dev/null"
    n_trees = 300
    X_train, X_test = pre_process(dataset_paths[name])
    seed = 42
    n_runs_imps = 2

    EDIFFI_original = Extended_DIFFI_original(
        n_trees=n_trees, max_depth=100, subsample_size=256, plus=1, seed=seed
    )

    imps_original = compute_imps(EDIFFI_original, X_train, X_test, n_runs_imps)

    # if parallel:
    #     print("Set up Extended_DIFFI_parallel")
    #     EDIFFI = Extended_DIFFI_parallel(
    #         n_trees=n_trees, max_depth=100, subsample_size=256, plus=1, seed=seed
    #     )
    #     EDIFFI.set_num_processes(n_cores_fit, n_cores_importance, n_cores_anomaly)
    #     print("Finished setting up Extended_DIFFI_parallel")
    # else:
