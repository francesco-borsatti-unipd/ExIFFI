import os
import numpy as np
from typing import Optional

import pandas as pd
from glob import glob


def load_stats(results_dirpath, filepath: Optional[str] = None, use_pkl: bool = False):
    """
    Load the results of the experiments in format of .npz files
    """

    if use_pkl:
        file_list = (
            [filepath] if filepath else glob(os.path.join(results_dirpath, "*.pkl"))
        )

        assert len(file_list) > 0, f"No files found in path {results_dirpath}"

        df_list = []
        for df in file_list:
            df_list.append(pd.read_pickle(df))

        return pd.concat(df_list, axis=0, ignore_index=True)

    file_list = [filepath] if filepath else glob(os.path.join(results_dirpath, "*.npz"))

    stats = [dict(np.load(path, allow_pickle=True)) for path in file_list]

    columns_to_merge = ["arguments", "execution_time_stat", "memory_MB_stats"]

    new_stats = []
    for result in stats:
        tmp_dict = {}

        for k, v in result.items():
            if k in columns_to_merge:
                continue
            tmp_dict[k] = v

        for dict_name in columns_to_merge:
            if dict_name not in result:
                continue
            dict_data = result[dict_name].tolist()

            for k, v in dict_data.items():
                tmp_dict[k] = v

        new_stats.append(tmp_dict)

    return pd.DataFrame(new_stats)


def display_stats(df):
    from IPython.display import display

    df = df.drop(columns=["importances_matrix", "filename"], inplace=False)
    df.set_index("time", inplace=True)
    display(df)


def compute_cpu_efficiency(real_time, user_time, n_cores):
    """
    Compute the efficiency of parallelization
    """
    return 100 * user_time / (real_time * n_cores)
