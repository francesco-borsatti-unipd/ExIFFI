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

    columns_to_merge = ["arguments", "execution_time_stat"]

    new_stats = []
    for result in stats:

        tmp_dict = {}

        for k, v in result.items():
            if k in columns_to_merge:
                continue
            tmp_dict[k] = v

        for dict_name in columns_to_merge:
            dict_data = result[dict_name].tolist()

            for k, v in dict_data.items():
                tmp_dict[k] = v
                
        new_stats.append(tmp_dict)

    return pd.DataFrame(new_stats)


def display_stats(df):
    df = df.drop(columns=["importances_matrix"], inplace=False)

    display(df)
