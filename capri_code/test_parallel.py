import sys
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


def load_data(filename):
    data = loadmat(mat_file_names_real[filename])
    X, y = data["X"], data["y"]
    y = np.hstack(y)
    return X, y


def compute_imps(model, X_train, X_test, n_runs, name, pwd, dim, f=6):
    name = "GFI_" + name

    # X_test=np.r_[X_train,X_test]

    imps = np.zeros(shape=(n_runs, X_train.shape[1]))
    for i in tqdm(range(n_runs)):
        model.fit(X_train)
        imps[i, :] = model.Global_importance(
            X_test, calculate=True, overwrite=False, depth_based=False
        )

    path = pwd + "/results/imp/imp_score_" + name + ".pkl"
    with open(path, "wb") as fl:
        pickle.dump(imps, fl)

    return imps


def parse_arguments():
    parser = argparse.ArgumentParser()
    # inputs and outputs
    parser.add_argument(
        "--savedir", type=str, required=True, help="Save directory for the results"
    )
    parser.add_argument("--parallel", action="store_true",help="Boolean to switch between parallel and serial code")
    parser.add_argument("--seed", type=int, default=None,help='Set seed for reproducibility')
    parser.add_argument("--n_runs", type=int, default=10,help='Set numner of runs')
    parser.add_argument("--n_cores", type=int, default=2,help='Set number of cores to use in parallel code')
    parser.add_argument("--num_trees", type=int, default=300,help='Number of trees in ExIFFI')
    return parser.parse_args()


def test_exiffi(
    X_train,
    X_test,
    X,
    savedir,
    n_runs=10,
    seed=None,
    parallel=False,
    n_cores=2,
    num_trees=300,
):
    args_to_avoid = ["X_train", "X_test", "X", "savedir"]
    args = dict()
    for k, v in locals().items():
        if k in args_to_avoid:
            continue
        args[k] = v

    ex_time = []
    ex_imps = {}

    for i in trange(n_runs):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = np.zeros(X_train.shape[0])
        y_test = np.ones(X_test.shape[0])
        y = np.concatenate([y_train, y_test])
        X_test = np.r_[X_train, X_test]
        scaler2 = StandardScaler()
        X = scaler2.fit_transform(X)

        seed = None if seed is None else seed + i

        if parallel:
            EDIFFI = Extended_DIFFI_parallel(
                n_trees=num_trees, max_depth=100, subsample_size=256, plus=1, seed=seed
            )
            EDIFFI.set_num_processes(n_cores, n_cores)
        else:
            EDIFFI = Extended_DIFFI_original(
                n_trees=num_trees, max_depth=100, subsample_size=256, plus=1, seed=seed
            )

        dim = X.shape[1]
        pwd = os.path.dirname(os.getcwd())
        start = time.time()
        imps = compute_imps(EDIFFI, X, X, 10, name, pwd, dim, f=6)
        ex_imps["Execution " + str(i)] = imps
        end = time.time()
        ex_time.append(end - start)

    # print(ex_imps)
    time_stat = {"mean": np.mean(ex_time), "std": np.std(ex_time)}
    filename = "test_stat_parallel.npz" if parallel else "test_stat_serial.npz"
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    filename = current_time + "_" + filename

    # if dir does not exist, create it
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filepath = os.path.join(savedir, filename)

    np.savez(
        filepath,
        execution_time_stat=time_stat,
        importances_matrix=ex_imps,
        arguments=args,
    )


if __name__ == "__main__":
    print("Testing Extended DIFFI")
    args = parse_arguments()
    name = "wine"
    path = os.getcwd()
    path = os.path.dirname(path)
    path_real = os.path.join(path, "data", "real")
    mat_files_real = glob(os.path.join(path_real, "*.mat"))
    mat_file_names_real = {os.path.basename(x).split(".")[0]: x for x in mat_files_real}
    X, y = load_data(name)
    X_train, X_test = partition_data(X, y)
    # print(f'X_train shape: {X_train.shape}')
    # print(f'X_test shape: {X_test.shape}')
    # print(f'X_train: {X_train}')
    # print(f'X_test: {X_test}')
    # print(f'n_runs: {args.n_runs}')
    # print(f'seed: {args.seed}')
    print(f"parallel: {args.parallel}")
    test_exiffi(
        X_train=X_train,
        X_test=X_test,
        X=X_test,
        savedir=args.savedir,
        n_runs=args.n_runs,
        seed=args.seed,
        parallel=args.parallel,
        n_cores=args.n_cores,
        num_trees=args.num_trees,
    )
