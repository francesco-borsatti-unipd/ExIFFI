import os
import numpy as np

import pandas as pd
from glob import glob
from IPython.display import display


def load_stats(results_dirpath):
    """
    Load the results of the experiments in format of .npz files
    """

    file_list = glob(os.path.join(results_dirpath, "*.npz"))
    stats = [dict(np.load(path, allow_pickle=True)) for path in file_list]

    new_stats = []
    for res in stats:
        tmp_dict = {}
        for k, v in res.items():
            if k in ["arguments", "execution_time_stat"]:
                continue
            tmp_dict[k] = v

        args = res["arguments"].tolist()

        for k, v in args.items():
            tmp_dict[k] = v

        execution_time_stat = res["execution_time_stat"].tolist()

        for k, v in execution_time_stat.items():
            tmp_dict[k + "_exec_time"] = v

        new_stats.append(tmp_dict)

    return pd.DataFrame(new_stats)


def display_stats(df):
    df = df.drop(columns=["importances_matrix"], inplace=False)

    display(df)
