import os, psutil

import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from append_dir import append_dirname

append_dirname("ExIFFI")
from utils.utils import partition_data
from utils.feature_selection import *

# from plot import *
# from simulation_setup import *
from models import *
from models.Extended_IF import *
from models.Extended_DIFFI_parallel import *
from models.Extended_DIFFI_original import *
import math
import seaborn as sns

sns.set()

from sklearn.preprocessing import StandardScaler
import time

import os
import pickle
from scipy.io import loadmat
from glob import glob

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


def drop_duplicates(X, y):
    """
    Drop duplicate rows from a dataset
    --------------------------------------------------------------------------------

    Parameters
    ----------
    X :         pd.DataFrame
        Input dataset
    y:          np.array
        Dataset labels

    Returns
    -------
    X,y:      Updated dataset and labels after the duplicates removal
    """

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
    """
    Upload a dataset from a .csv file. This function was used for the Diabetes and Moodify datasets.
    --------------------------------------------------------------------------------

    Parameters
    ----------
    name :         string
        Dataset's name
    path:          string
        Path of the .csv file containing the dataset

    Returns
    -------
    X,y:      X contains the dataset input features as a pd.DataFrame while y contains the dataset's labels as a np.array
    """

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

    return X_train, X_test


def compute_imps(model, X_train, X_test, n_runs):
    print(f"shape of X_train: {X_train.shape}")
    print(f"shape of X_test: {X_test.shape}")
    mem_MB_lst = []

    imps = np.zeros(shape=(n_runs, X_train.shape[1]))
    for i in trange(n_runs, desc="Fit & Importances"):
        print('Start fit')
        model.fit(X_train)
        print('End fit')
        print('Start Global Importance')
        imps[i, :] = model.Global_importance(
            X_test, calculate=True, overwrite=False, depth_based=False
        )
        print('End Global Importance')
        mem_MB_lst.append(psutil.Process(os.getpid()).memory_info().rss / 1000**2)

    return imps, mem_MB_lst


def parse_arguments():
    parser = argparse.ArgumentParser()
    # inputs and outputs
    parser.add_argument(
        "--savedir", type=str, required=True, help="Save directory for the results"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set seed for reproducibility"
    )
    parser.add_argument("--n_runs", type=int, default=10, help="Set numner of runs")
    parser.add_argument(
        "--n_cores",
        type=int,
        default=1,
        help="Set number of cores to use in parallel code. If 1 the code is serial, otherwise it is parallel",
    )
    parser.add_argument(
        "--n_runs_imps",
        type=int,
        default=10,
        help="Set number of runs for the importance computation",
    )
    parser.add_argument(
        "--n_trees", type=int, default=300, help="Number of trees in ExIFFI"
    )
    parser.add_argument(
        "--dataset_names",
        nargs="+",
        type=str,
        default=[],
        help="List of names of datasets to test ExIFFI on",
    )
    return parser.parse_args()


def test_exiffi(
    X_train,
    X_test,
    savedir,
    n_runs=10,
    seed=None,
    parallel=False,
    n_cores=2,
    n_trees=300,
    name="",
    n_runs_imps = 10,
):
    args_to_avoid = ["X_train", "X_test", "savedir", "args_to_avoid", "args"]
    args = dict()
    for k, v in locals().items():
        if k in args_to_avoid:
            continue
        args[k] = v

    ex_time = []
    ex_imps = []  # list of importance matrices for one execution
    ex_mem_MB = []

    for i in trange(n_runs, desc="Experiment"):
        print('Execution 1')
        start = time.time()

        seed = None if seed is None else seed + i * n_trees

        if parallel:
            print('Set up Extended_DIFFI_parallel')
            EDIFFI = Extended_DIFFI_parallel(
                n_trees=n_trees, max_depth=100, subsample_size=256, plus=1, seed=seed
            )
            EDIFFI.set_num_processes(n_cores, n_cores)
            print('Finished setting up Extended_DIFFI_parallel')
        else:
            EDIFFI = Extended_DIFFI_original(
                n_trees=n_trees, max_depth=100, subsample_size=256, plus=1, seed=seed
            )

        print('Call compute_imps')
        imps, mem_MB = compute_imps(EDIFFI, X_train, X_test, n_runs_imps)
        print('End call compute_imps')
        ex_imps.append(imps)
        ex_mem_MB.append(mem_MB)

        ex_time.append(time.time() - start)

    # print(ex_imps)
    time_stat = {"mean_time": np.mean(ex_time), "std_time": np.std(ex_time)}
    mem_stat = {
        "mean_MB": np.mean(ex_mem_MB),
        "std_MB": np.std(ex_mem_MB),
        "max_MB": np.max(ex_mem_MB),
    }
    filename = "test_stat_parallel.npz" if parallel else "test_stat_serial.npz"
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    filename = current_time + "_" + name + "_" + filename

    # if dir does not exist, create it
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filepath = os.path.join(savedir, filename)

    np.savez(
        filepath,
        execution_time_stat=time_stat,
        importances_matrix=ex_imps,
        memory_MB_stats=mem_stat,
        arguments=args,
        time=pd.Timestamp.now(),
    )


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

    args.parallel = args.n_cores > 1

    print("#" * 60)
    print(f"TESTING {'PARALLEL' if args.parallel else 'SERIAL'} ExIFFI")
    print("#" * 60)
    print("TEST PARAMETERS:")
    print(f"Number of runs: {args.n_runs}")
    print(f"Number of trees: {args.n_trees}")
    print(f"Number of cores: {args.n_cores}")
    print(f"Seed: {args.seed}")
    print(f"Parallel: {args.parallel}")
    print("#" * 60)

    if len(args.dataset_names) > 0:
        dataset_names = args.dataset_names
    else:
        dataset_names = sorted(dataset_names)

    print("dataset_names", dataset_names)

    for name in dataset_names:
        print("#" * 60)
        print(f"DATASET: {name}")
        print("#" * 60)
        X_train, X_test = pre_process(dataset_paths[name])
        test_exiffi(
            X_train=X_train,
            X_test=X_test,
            savedir=args.savedir,
            n_runs=args.n_runs,
            seed=args.seed,
            parallel=args.parallel,
            n_cores=args.n_cores,
            n_trees=args.n_trees,
            name=name,
            n_runs_imps=args.n_runs_imps,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
