import os, psutil
import time
import warnings

import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from scipy.io import loadmat
from glob import glob
from sklearn.preprocessing import StandardScaler

# import seaborn as sns
# sns.set()


from append_dir import append_dirname

append_dirname("ExIFFI")
from utils.utils import partition_data
from utils.feature_selection import *
from process_results import load_stats

# from models import *

from models.Extended_DIFFI_parallel import Extended_DIFFI_parallel
from models.Extended_DIFFI_original import Extended_DIFFI_original
from models.Extended_DIFFI_C import Extended_DIFFI_c


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

    return X_train.astype(np.float64), X_test.astype(np.float64)


def compute_imps(model, X_train, X_test, n_runs):
    print(f"shape of X_train: {X_train.shape}")
    print(f"shape of X_test: {X_test.shape}")
    mem_MB_lst = []

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
    parser.add_argument("--n_runs", type=int, default=10, help="Set number of runs")
    parser.add_argument(
        "--n_cores",
        type=int,
        nargs="+",
        default=[1],
        help="Set number of cores to use. "
        + "If [1] the code is serial, otherwise it is parallel. "
        + "List of 1 or 3 integers, respectively num processes of fit, importance and anomaly",
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
        required=True,
        nargs="+",
        type=str,
        help="List of names of datasets to test ExIFFI on",
    )
    parser.add_argument(
        "--wrapper",
        action="store_true",
        help="If set, run the wrapper for timing the code",
    )
    parser.add_argument(
        "--add_bash",
        action="store_true",
        help="If set, add bash -c to the command for timing the code",
    )
    parser.add_argument(
        "--use_C",
        action="store_true",
        help="If set, use the C implementation of the Extended_DIFFI",
    )
    # parser.add_argument(
    #     "--n_threads",
    #     type=int,
    #     default=12,
    #     help="Number of threads to use in the C implementation"
    # )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename of the output saved file. If None, it is automatically generated",
    )
    return parser.parse_args()


def test_exiffi(
    X_train,
    X_test,
    savedir: str,
    n_cores_fit,
    n_cores_importance,
    n_cores_anomaly,
    n_runs=10,
    seed=None,
    parallel=False,
    n_trees=300,
    name="",
    n_runs_imps=10,
    filename=None,
    use_C=False,
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
        print("Execution 1")
        start = time.time()

        if seed is not None:
            seed += 1
            np.random.seed(seed)

        if use_C:
            print("Set up Extended_DIFFI_C")
            EDIFFI = Extended_DIFFI_c(
                n_trees=n_trees,
                max_depth=8,
                subsample_size=256,
                plus=1,
            )
            #EDIFFI.set_num_threads()
        elif parallel:
            print("Set up Extended_DIFFI_parallel")
            EDIFFI = Extended_DIFFI_parallel(
                n_trees=n_trees, max_depth=8, subsample_size=256, plus=1
            )
            EDIFFI.set_num_processes(n_cores_fit, n_cores_importance, n_cores_anomaly)
            print("Finished setting up Extended_DIFFI_parallel")
        else:
            print("Set up Extended_DIFFI_original")
            EDIFFI = Extended_DIFFI_original(
                n_trees=n_trees, max_depth=8, subsample_size=256, plus=1
            )

        print("Call compute_imps")
        imps, mem_MB = compute_imps(EDIFFI, X_train, X_test, n_runs_imps)
        print("End call compute_imps")
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

    if filename is None:
        filename = get_filename(parallel, name)

    # if dir does not exist, create it
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filepath = os.path.join(savedir, filename)

    np.savez(
        filepath,
        execution_time_stat=time_stat,  # type: ignore
        importances_matrix=ex_imps,
        memory_MB_stats=mem_stat,  # type: ignore
        arguments=args,  # type: ignore
        time=pd.Timestamp.now(),  # type: ignore
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

    print("#" * 60)
    print(f"TESTING {'PARALLEL' if args.parallel else 'SERIAL'} ExIFFI")
    print("#" * 60)
    print("TEST PARAMETERS:")
    print(f"Number of runs: {args.n_runs}")
    print(f"Number of trees: {args.n_trees}")
    print(
        f"Number of cores: fit {n_cores_fit}, importance {n_cores_importance}, anomaly {n_cores_anomaly}"
    )
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
            n_trees=args.n_trees,
            name=name,
            n_runs_imps=args.n_runs_imps,
            n_cores_fit=n_cores_fit,
            n_cores_importance=n_cores_importance,
            n_cores_anomaly=n_cores_anomaly,
            filename=args.filename,
            use_C=args.use_C,
        )


def str_to_sec(time_str):
    """
    Takes a string of the form "1m5,928s" and returns the time in seconds
    """
    time_str = time_str.replace(",", ".")
    time_str = time_str.replace("s", "")
    time_str = time_str.split("m")
    time_sec = float(time_str[0]) * 60 + float(time_str[1])
    return time_sec


def get_filename(is_parallel: bool, dataset_name: str):
    partial_filename = "test_stat_parallel" if is_parallel else "test_stat_serial"
    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    partial_filename = (
        current_time + "_" + partial_filename + "_" + dataset_name + ".npz"
    )
    return partial_filename


if __name__ == "__main__":
    args = parse_arguments()

    if args.wrapper:
        import subprocess

        arg_str = sys.argv[1:]

        # remove "--wrapper" from the list of arguments
        arg_str.remove("--wrapper")

        # remove the "--dataset_names" argument and its value from the list of arguments
        arg_str.remove("--dataset_names")
        for dataset_name in args.dataset_names:
            arg_str.remove(dataset_name)

        args.parallel = any(n > 1 for n in args.n_cores)

        # if more than one dataset name is given, remove them from the list of arguments
        # and iterate one at a time, adding just one dataset name to the arguments

        # dobbiamo usare il name completo dato che ne facciamo uno alla volta

        for dataset_name in args.dataset_names:
            filename = get_filename(args.parallel, dataset_name)

            whole_command = (
                ["time", "python", "test_parallel.py"]
                + arg_str
                + ["--dataset_names", dataset_name]
                + ["--filename", filename]
            )
            whole_command = " ".join(whole_command)

            if args.add_bash:
                whole_command = f'bash -c "{whole_command}"'

            print("\n\nwhole command\n", whole_command)

            output = subprocess.check_output(
                whole_command, stderr=subprocess.STDOUT, text=True, shell=True
            )

            linux_time_stats = [i.split("\t") for i in output.split("\n")[-4:-1]]

            print("\n\noutput\n", output)

            # now "linux_time_stats" needs to be added to the dataframe
            # --> list the files in the savedir with glob
            # --> use partial_filename to find the npz
            # --> find the npz name (or names if multiple datasets)
            # --> load the npz with the "process_results.py" script
            # --> add the stats to the dataframe
            # --> save the dataframe as pkl with df.to_pickle() with the same name as the npz

            filepath = os.path.join(args.savedir, filename)

            stats = load_stats(args.savedir, filepath=filepath)

            for ts in linux_time_stats:
                stats[ts[0] + "_time"] = str_to_sec(ts[1])

            print(stats)

            print("\n\nSaving time stats in", filepath)

            stats.to_pickle(filepath.replace(".npz", ".pkl"))

            # delete the npz file
            os.remove(filepath)

    else:
        main(args)
