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
    stats = [np.load(path, allow_pickle=True) for path in file_list]
    stats = {data["arguments"].tolist()["name"]: dict(data) for data in stats}

    new_stats = []
    for _, value in stats.items():
        tmp_dict = {}
        for k, v in value.items():
            if k in ["arguments", "execution_time_stat"]:
                continue
            tmp_dict[k] = v

        args = value["arguments"].tolist()

        for k, v in args.items():
            tmp_dict[k] = v

        execution_time_stat = value["execution_time_stat"].tolist()

        for k, v in execution_time_stat.items():
            tmp_dict[k + "_exec_time"] = v

        new_stats.append(tmp_dict)

    return pd.DataFrame(new_stats)


def display_stats(df):
    df = df.drop(columns=["importances_matrix"], inplace=False)

    # set "name" as the index
    df.set_index("name", inplace=True)

    display(df)
