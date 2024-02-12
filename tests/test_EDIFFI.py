import os, sys, timeit, logging


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

import numpy as np
import pandas as pd
from tqdm import trange
from scipy.io import loadmat
from glob import glob
from sklearn.preprocessing import StandardScaler

from utils.utils import partition_data
from utils.feature_selection import *

from models import *
from models.Extended_IF import *
from models.Extended_DIFFI_parallel import Extended_DIFFI_parallel
from models.Extended_DIFFI_original import Extended_DIFFI_original
from models.Extended_DIFFI_C import Extended_DIFFI_c


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


def compute_imps(model, X_train, X_test, n_runs):

    imps = np.zeros(shape=(n_runs, X_train.shape[1]))
    for i in trange(n_runs, desc="Fit & Importances", disable=True):
        model.fit(X_train)
        print("Start Global Importance")
        imps[i, :] = model.Global_importance(
            X_test, calculate=True, overwrite=False, depth_based=False
        )
        print("End Global Importance")

    return imps


def test_exiffi(
    X_train,
    X_test,
    n_cores_fit,
    n_cores_importance,
    n_cores_anomaly,
    seed=None,
    parallel=False,
    use_c=False,
    n_trees=300,
    n_runs_imps=10,
):
    if seed is not None:
        np.random.seed(seed)

    max_depth = 100
    subsample_size = 256

    if use_c:
        print("... Initialize Extended_DIFFI_c ...")
        EDIFFI = Extended_DIFFI_c(
            n_trees=n_trees, max_depth=max_depth, subsample_size=subsample_size, plus=1
        )
    elif parallel:
        print("... Initialize Extended_DIFFI_parallel ...")
        EDIFFI = Extended_DIFFI_parallel(
            n_trees=n_trees, max_depth=max_depth, subsample_size=subsample_size, plus=1
        )
        EDIFFI.set_num_processes(n_cores_fit, n_cores_importance, n_cores_anomaly)
    else:
        print("... Initialize Extended_DIFFI_serial ...")
        EDIFFI = Extended_DIFFI_original(
            n_trees=n_trees, max_depth=max_depth, subsample_size=subsample_size, plus=1
        )

    print("Computing imps")
    compute_imps(EDIFFI, X_train, X_test, n_runs_imps)


def main(args):
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

    if len(args.n_cores) == 1:
        n_cores_fit = args.n_cores[0]
        n_cores_importance = args.n_cores[0]
        n_cores_anomaly = args.n_cores[0]
    elif len(args.n_cores) == 3:
        n_cores_fit = args.n_cores[0]
        n_cores_importance = args.n_cores[1]
        n_cores_anomaly = args.n_cores[2]
    else:
        raise ValueError("Number of elements in --n_cores must be either 1 or 3")

    args.parallel = any(n > 1 for n in args.n_cores)

    print(f"num_trees: {args.n_trees}")
    print(
        f"num_cpus: fit {n_cores_fit}, imp {n_cores_importance}, anomaly {n_cores_anomaly}"
    )

    if len(args.dataset_names) > 0:
        dataset_names = args.dataset_names
    else:
        dataset_names = sorted(dataset_names)

    print("dataset_names", dataset_names)

    for name in dataset_names:
        print(f"Dataset: {name}")
        X_train, X_test = pre_process(dataset_paths[name])
        test_exiffi(
            X_train=X_train,
            X_test=X_test,
            seed=args.seed,
            n_trees=args.n_trees,
            n_runs_imps=args.n_runs_imps,
            n_cores_fit=n_cores_fit,
            n_cores_importance=n_cores_importance,
            n_cores_anomaly=n_cores_anomaly,
            use_c=args.use_c,
            parallel=args.parallel,
        )


if __name__ == "__main__":

    class Args:
        seed = 42
        parallel = True
        n_trees = 100
        n_runs_imps = 1
        n_cores = [8]  # fit, importance, anomaly
        n_cores_fit = None
        n_cores_importance = None
        n_cores_anomaly = None
        dataset_names = ["wine"]
        use_c = False
        parallel = True

    print("... Start timing tests ...")
    num_runs = 1
    args = Args()

    python_time = timeit.timeit(lambda: main(args), number=num_runs)

    print("-"*100)
    args.use_c = True
    c_time = timeit.timeit(lambda: main(args), number=num_runs)

    print("-"*100)
    print(f"Python time: {python_time/num_runs}")
    print(f"C time: {c_time/num_runs}")
    print(f"C speedup: {100*python_time/c_time:.2f}", r"%")
