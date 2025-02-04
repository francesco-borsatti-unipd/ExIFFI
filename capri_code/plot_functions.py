from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import seaborn as sns

sns.set()


def get_all_stats_dataset(data, name):
    df = data.groupby(["name"]).get_group((name))
    return df


def df_for_plots(data, name):
    df = get_all_stats_dataset(data, name)
    df = df[
        [
            "name",
            "n_trees",
            "real_time",
            "user_time",
            "cpu_efficiency",
            "max_MB",
            "real_time_single_run",
            "n_cores_fit",
            "n_cores_importance",
            "n_cores_anomaly",
        ]
    ]
    df["n_cores"] = [
        np.max(np.array([i, j, k]))
        for i, j, k in zip(
            df["n_cores_fit"], df["n_cores_importance"], df["n_cores_anomaly"]
        )
    ]
    df.drop(
        columns=["n_cores_fit", "n_cores_importance", "n_cores_anomaly"], inplace=True
    )
    df.sort_values(by="cpu_efficiency", ascending=False)
    return df


def barplot(
    name: str, df: pd.DataFrame, x_col: str, y_col: str, ylim_off: int = 0, color=None
):

    ax = sns.barplot(x=x_col, y=y_col, data=df, width=0.3, color=color, saturation=0.5)
    ylim = df[y_col].max()
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(name, loc="left", color="red")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    plt.ylim(0, ylim + ylim_off)
    return ax


def barplot_subplots(
    ax,
    name,
    df,
    x_col,
    y_col,
    ylim_off=0,
    remove_xticks=False,
    rotate_xticks=False,
    color=None,
    remove_patches=False,
    title_loc="left",
    bar_width=80,
):
    if color is not None:
        # ax = sns.barplot(x=x_col, y=y_col, data=df, width=0.3, ax=ax, color=color)
        ax.bar(df[x_col], df[y_col], color=color, width=bar_width)
    else:
        # ax.bar(df[x_col], df[y_col], width=80)
        ax = sns.barplot(x=x_col, y=y_col, data=df, width=0.3, ax=ax)

    ylim = df[y_col].max()
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(name, loc=title_loc, color="red")

    if not remove_patches:
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
    # set y log scale
    # ax.set_yscale('log')

    if remove_xticks:
        ax.set_xticks([])
    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylim(0, ylim + ylim_off)
    return ax


def plot_bars_side_by_side(
    name: str,
    df: pd.DataFrame,
    df_fast: pd.DataFrame,
    num_cores,
    n_trees,
    plotsize: tuple,
    ylim_off: float = 0.0,
    x_offset: float = 0.0,
    labels=None,
):
    # df, df_fast
    grid = namedtuple("Grid", ["rows", "cols"])(1, len(num_cores))

    dfs_norm = []
    dfs_fast = []

    for num_core in num_cores:
        df_100 = df.groupby(["n_trees", "n_cores"]).get_group((100, num_core))
        df_300 = df.groupby(["n_trees", "n_cores"]).get_group((300, num_core))
        df_600 = df.groupby(["n_trees", "n_cores"]).get_group((600, num_core))
        dfs_norm.append(
            pd.DataFrame(
                {
                    "n_trees": np.array(n_trees) - x_offset,
                    "real_time_mean": [
                        np.mean(df_100["real_time_single_run"]),
                        np.mean(df_300["real_time_single_run"]),
                        np.mean(df_600["real_time_single_run"]),
                    ],
                }
            )
        )
        df_100 = df_fast.groupby(["n_trees", "n_cores"]).get_group((100, num_core))
        df_300 = df_fast.groupby(["n_trees", "n_cores"]).get_group((300, num_core))
        df_600 = df_fast.groupby(["n_trees", "n_cores"]).get_group((600, num_core))
        dfs_fast.append(
            pd.DataFrame(
                {
                    "n_trees": np.array(n_trees) + x_offset,
                    "real_time_mean": [
                        np.mean(df_100["real_time_single_run"]),
                        np.mean(df_300["real_time_single_run"]),
                        np.mean(df_600["real_time_single_run"]),
                    ],
                }
            )
        )

    fig, axes = plt.subplots(grid.rows, grid.cols, figsize=plotsize)
    names = [
        f"{name} 1 core",
        f"{name} 4 cores",
        f"{name} 8 cores",
        f"{name} 12 cores",
        f"{name} 16 cores",
    ]

    for i, (ax, exp_name, df_norm, df_fast) in enumerate(
        zip(axes.flatten(), names, dfs_norm, dfs_fast)
    ):
        barplot_subplots(
            ax,
            exp_name,
            df_norm,
            "n_trees",
            "real_time_mean",
            ylim_off=ylim_off,
            color="blue",
            remove_xticks=True,
            remove_patches=True,
        )
        a = barplot_subplots(
            ax,
            exp_name,
            df_fast,
            "n_trees",
            "real_time_mean",
            ylim_off=ylim_off,
            color="red",
            remove_xticks=True,
            remove_patches=True,
        )
        ax.set_xticks(n_trees)

    if labels is not None:
        a.plot([], [], color="red", label=labels[0])
        a.plot([], [], color="blue", label=labels[1])
    else:
        a.plot([], [], color="red", label="Fast")
        a.plot([], [], color="blue", label="Normal")

    a.legend(loc="upper right", framealpha=1, facecolor="white")
    plt.tight_layout()
    plt.show()


def plot_bars_side_by_side_cores(
    name: str,
    df: pd.DataFrame,
    df_fast: pd.DataFrame,
    num_cores,
    n_trees,
    plotsize: tuple,
    ylim_off: float = 0.0,
    x_offset: float = 0.0,
    labels=None,
    ylog=False,
):

    grid = namedtuple("Grid", ["rows", "cols"])(1, len(n_trees))

    dfs_norm = []
    dfs_fast = []

    for num_tree in n_trees:
        df_1 = df.groupby(["n_cores", "n_trees"]).get_group((1, num_tree))
        df_4 = df.groupby(["n_cores", "n_trees"]).get_group((4, num_tree))
        df_8 = df.groupby(["n_cores", "n_trees"]).get_group((8, num_tree))
        df_12 = df.groupby(["n_cores", "n_trees"]).get_group((12, num_tree))
        dfs_norm.append(
            pd.DataFrame(
                {
                    "n_cores": np.array(num_cores) - x_offset,
                    "real_time_mean": [
                        np.mean(df_1["real_time_single_run"]),
                        np.mean(df_4["real_time_single_run"]),
                        np.mean(df_8["real_time_single_run"]),
                        np.mean(df_12["real_time_single_run"]),
                    ],
                }
            )
        )
        df_1 = df_fast.groupby(["n_cores", "n_trees"]).get_group((1, num_tree))
        df_4 = df_fast.groupby(["n_cores", "n_trees"]).get_group((4, num_tree))
        df_8 = df_fast.groupby(["n_cores", "n_trees"]).get_group((8, num_tree))
        df_12 = df_fast.groupby(["n_cores", "n_trees"]).get_group((12, num_tree))
        dfs_fast.append(
            pd.DataFrame(
                {
                    "n_cores": np.array(num_cores) + x_offset,
                    "real_time_mean": [
                        np.mean(df_1["real_time_single_run"]),
                        np.mean(df_4["real_time_single_run"]),
                        np.mean(df_8["real_time_single_run"]),
                        np.mean(df_12["real_time_single_run"]),
                    ],
                }
            )
        )

    fig, axes = plt.subplots(grid.rows, grid.cols, figsize=plotsize, sharey=True)
    names = [f"{name} 100 trees", f"{name} 300 trees", f"{name} 600 trees"]

    for i, (ax, exp_name, df_norm, df_fast) in enumerate(
        zip(axes.flatten(), names, dfs_norm, dfs_fast)
    ):
        barplot_subplots(
            ax,
            exp_name,
            df_norm,
            "n_cores",
            "real_time_mean",
            ylim_off=ylim_off,
            color="blue",
            remove_xticks=True,
            remove_patches=True,
            bar_width=1,
        )
        a = barplot_subplots(
            ax,
            exp_name,
            df_fast,
            "n_cores",
            "real_time_mean",
            ylim_off=ylim_off,
            color="red",
            remove_xticks=True,
            remove_patches=True,
            bar_width=1,
        )
        ax.set_xticks(num_cores)

    if ylog:
        set_y_and_x(
            axes, dfs_norm, col_to_use="real_time_mean", x_ticks=num_cores, ylim_off=ylim_off
        )

    if labels is not None:
        a.plot([], [], color="red", label=labels[0])
        a.plot([], [], color="blue", label=labels[1])
    else:
        a.plot([], [], color="red", label="Fast")
        a.plot([], [], color="blue", label="Normal")

    a.legend(loc="upper right", framealpha=1, facecolor="white")
    plt.tight_layout()
    plt.show()


def multiple_time_tree_plot(
    name, df, n_cores=[1, 4, 8, 12, 16], n_trees=[100, 300, 600], ylim_off=10
):
    """
    Plot n_trees vs real_time for different number of cores
    """
    dfs_plot = []
    for num_core in n_cores:
        df_100 = df.groupby(["n_trees", "n_cores"]).get_group((100, num_core))
        df_300 = df.groupby(["n_trees", "n_cores"]).get_group((300, num_core))
        df_600 = df.groupby(["n_trees", "n_cores"]).get_group((600, num_core))
        dfs_plot.append(
            pd.DataFrame(
                {
                    "n_trees": n_trees,
                    "real_time_mean": [
                        np.mean(df_100["real_time_single_run"]),
                        np.mean(df_300["real_time_single_run"]),
                        np.mean(df_600["real_time_single_run"]),
                    ],
                }
            )
        )

    # fig, axes = plt.subplots(1,1,figsize=(10, 8))
    dfs = dfs_plot
    names = [
        f"{name} 1 core",
        f"{name} 4 cores",
        f"{name} 8 cores",
        f"{name} 12 cores",
        f"{name} 16 cores",
    ]
    colors = ["r", "g", "b", "y", "c"]
    for name, df, c in zip(names, dfs, colors):
        barplot(name, df, "n_trees", "real_time_mean", ylim_off=ylim_off, color=c)
        # for p in ax.patches:
        #     ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
        #             ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # handles, labels = axes.get_legend_handles_labels()
    # axes.legend(handles=handles, labels=['1 core', '4 cores', '8 cores', '12 cores', '16 cores'])
    # plt.tight_layout()
    plt.ylim(0, ylim_off)
    plt.show()


def set_y_and_x(axes, dfs, col_to_use, x_ticks, ylim_off=0):
    # get the max y value
    y_max = max([df[col_to_use].max() for df in dfs])
    # create a list of the yticks
    y_ticks = np.logspace(0, np.log10(y_max), num=5)

    for ax in axes.flatten():
        ax.set_yscale("log")
        ax.set_ylim(0, y_max + ylim_off)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(y)}" for y in y_ticks])
        break

    # for ax in axes.flatten():
    #    ax.set_xticks(x_ticks)


def time_tree_plot(
    data,
    name,
    df,
    grid=(1, 5),
    plotsize=(15, 10),
    n_cores=[1, 4, 8, 12, 16],
    n_trees=[100, 300, 600],
    ylim_off=0,
    remove_yticks=False,
    title_loc="left",
):
    dfs_plot = []
    for num_core in n_cores:
        df_100 = df.groupby(["n_trees", "n_cores"]).get_group((100, num_core))
        df_300 = df.groupby(["n_trees", "n_cores"]).get_group((300, num_core))
        df_600 = df.groupby(["n_trees", "n_cores"]).get_group((600, num_core))
        dfs_plot.append(
            pd.DataFrame(
                {
                    "n_trees": n_trees,
                    "real_time_mean": [
                        np.mean(df_100["real_time_single_run"]),
                        np.mean(df_300["real_time_single_run"]),
                        np.mean(df_600["real_time_single_run"]),
                    ],
                }
            )
        )

    fig, axes = plt.subplots(grid[0], grid[1], figsize=plotsize, sharey=True)
    dfs = dfs_plot
    names = [
        f"{name} 1 core",
        f"{name} 4 cores",
        f"{name} 8 cores",
        f"{name} 12 cores",
        f"{name} 16 cores",
    ]

    for i, (ax, exp_name, df) in enumerate(zip(axes.flatten(), names, dfs)):
        barplot_subplots(
            ax,
            exp_name,
            df,
            "n_trees",
            "real_time_mean",
            ylim_off=ylim_off,
            title_loc=title_loc,
        )

    # dataframe=df_for_plots(data,name)
    # multiple_time_tree_plot(name,dataframe,n_cores=n_cores,n_trees=n_trees,ylim_off=ylim_off)

    if remove_yticks:
        plt.yticks([])

    set_y_and_x(
        axes, dfs, col_to_use="real_time_mean", x_ticks=n_trees, ylim_off=ylim_off
    )

    plt.tight_layout()
    plt.show()


"""
Plot n_cores vs real_time for different number of trees
"""


def time_core_plot(
    name,
    df,
    grid=(1, 3),
    plotsize=(15, 10),
    n_cores=[1, 4, 8, 12, 16],
    n_trees=[100, 300, 600],
    ylim_off=0,
    remove_yticks=False,
    no_16=False,
    title_loc="left",
):
    dfs_plot = []
    for num_tree in n_trees:
        df_1 = df.groupby(["n_cores", "n_trees"]).get_group((1, num_tree))
        df_4 = df.groupby(["n_cores", "n_trees"]).get_group((4, num_tree))
        df_8 = df.groupby(["n_cores", "n_trees"]).get_group((8, num_tree))
        df_12 = df.groupby(["n_cores", "n_trees"]).get_group((12, num_tree))
        if no_16:
            dfs_plot.append(
                pd.DataFrame(
                    {
                        "n_cores": n_cores,
                        "real_time_mean": [
                            np.mean(df_1["real_time_single_run"]),
                            np.mean(df_4["real_time_single_run"]),
                            np.mean(df_8["real_time_single_run"]),
                            np.mean(df_12["real_time_single_run"]),
                        ],
                    }
                )
            )
        else:
            df_16 = df.groupby(["n_cores", "n_trees"]).get_group((16, num_tree))
            dfs_plot.append(
                pd.DataFrame(
                    {
                        "n_cores": n_cores,
                        "real_time_mean": [
                            np.mean(df_1["real_time_single_run"]),
                            np.mean(df_4["real_time_single_run"]),
                            np.mean(df_8["real_time_single_run"]),
                            np.mean(df_12["real_time_single_run"]),
                            np.mean(df_16["real_time_single_run"]),
                        ],
                    }
                )
            )

    fig, axes = plt.subplots(grid[0], grid[1], figsize=plotsize, sharey=True)
    dfs = dfs_plot
    names = [f"{name} 100 trees", f"{name} 300 trees", f"{name} 600 trees"]

    for i, (ax, name, df) in enumerate(zip(axes.flatten(), names, dfs)):
        barplot_subplots(
            ax,
            name,
            df,
            "n_cores",
            "real_time_mean",
            ylim_off=ylim_off,
            title_loc=title_loc,
        )

    if remove_yticks:
        plt.yticks([])

    set_y_and_x(
        axes, dfs, col_to_use="real_time_mean", x_ticks=n_cores, ylim_off=ylim_off
    )

    plt.tight_layout()
    plt.show()


"""
Plot n_cores vs max_MB for different number of trees
"""


def time_mem_plot(
    name, df, n_cores=[1, 4, 8, 12, 16], n_trees=[100, 300, 600], ylim_off=10
):
    dfs_plot = []
    for num_tree in n_trees:
        df_4 = df.groupby(["n_cores", "n_trees"]).get_group((4, num_tree))
        df_8 = df.groupby(["n_cores", "n_trees"]).get_group((8, num_tree))
        df_12 = df.groupby(["n_cores", "n_trees"]).get_group((12, num_tree))
        df_16 = df.groupby(["n_cores", "n_trees"]).get_group((16, num_tree))
        dfs_plot.append(
            pd.DataFrame(
                {
                    "n_cores": n_cores,
                    "max_MB": [
                        df_4["max_MB"].max(),
                        df_8["max_MB"].max(),
                        df_12["max_MB"].max(),
                        df_16["max_MB"].max(),
                    ],
                }
            )
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    dfs = dfs_plot
    names = [f"{name} 100 trees", f"{name} 300 trees", f"{name} 600 trees"]

    for i, (ax, name, df) in enumerate(zip(axes.flatten(), names, dfs)):
        barplot_subplots(ax, name, df, "n_cores", "max_MB", ylim_off=ylim_off)

    plt.tight_layout()
    plt.show()


"""
Plot n_cores vs cpu_efficiency for different number of trees
"""


def time_eff_plot(
    name,
    df,
    grid=(1, 3),
    plotsize=(15, 10),
    n_cores=[1, 4, 8, 12, 16],
    n_trees=[100, 300, 600],
    ylim_off=0,
    remove_yticks=False,
    no_16=False,
    title_loc="left",
):
    dfs_plot = []
    for num_tree in n_trees:
        df_1 = df.groupby(["n_cores", "n_trees"]).get_group((1, num_tree))
        df_4 = df.groupby(["n_cores", "n_trees"]).get_group((4, num_tree))
        df_8 = df.groupby(["n_cores", "n_trees"]).get_group((8, num_tree))
        df_12 = df.groupby(["n_cores", "n_trees"]).get_group((12, num_tree))
        if no_16:
            dfs_plot.append(
                pd.DataFrame(
                    {
                        "n_cores": n_cores,
                        "cpu_efficiency": [
                            df_1["cpu_efficiency"].max(),
                            df_4["cpu_efficiency"].max(),
                            df_8["cpu_efficiency"].max(),
                            df_12["cpu_efficiency"].max(),
                        ],
                    }
                )
            )
        else:
            df_16 = df.groupby(["n_cores", "n_trees"]).get_group((16, num_tree))
            dfs_plot.append(
                pd.DataFrame(
                    {
                        "n_cores": n_cores,
                        "cpu_efficiency": [
                            df_1["cpu_efficiency"].max(),
                            df_4["cpu_efficiency"].max(),
                            df_8["cpu_efficiency"].max(),
                            df_12["cpu_efficiency"].max(),
                            df_16["cpu_efficiency"].max(),
                        ],
                    }
                )
            )

    fig, axes = plt.subplots(grid[0], grid[1], figsize=plotsize)
    dfs = dfs_plot
    names = [f"{name} 100 trees", f"{name} 300 trees", f"{name} 600 trees"]

    for i, (ax, name, df) in enumerate(zip(axes.flatten(), names, dfs)):
        barplot_subplots(
            ax,
            name,
            df,
            "n_cores",
            "cpu_efficiency",
            ylim_off=ylim_off,
            title_loc=title_loc,
        )

    if remove_yticks:
        plt.yticks([])

    plt.tight_layout()
    plt.show()


"""
Plot dataset names versus cpu efficiency for different number of cores 
"""


def df_cpu_eff_plot(data, n_cores, n_trees):
    # Group by n_cores and n_trees and sort values by cpu_efficiency
    df = data.groupby(["n_cores_anomaly", "n_trees"]).get_group((n_cores, n_trees))
    df.sort_values(by="cpu_efficiency", ascending=True, inplace=True)

    # Remove duplicate rows keeping the one with the highest cpu_efficiency
    df.sort_values(
        by=["name", "cpu_efficiency"], ascending=[False, False], inplace=True
    )
    df.drop_duplicates(subset=["name"], keep="first", inplace=True)

    # Reorder rows by increasing order of dataset complexity
    order = [
        "wine",
        "glass",
        "pima",
        "breastw",
        "ionosphere",
        "cardio",
        "annthyroid",
        "pendigits",
        "diabetes",
        "shuttle",
        "moodify",
    ]
    df["name"] = pd.Categorical(df["name"], categories=order, ordered=True)
    df.sort_values("name", inplace=True)

    # Obtain the barplot using barplot_subplots function
    # fig,ax=plt.subplots(1,1,figsize=(10,8))
    # ax=barplot_subplots(ax,f'CPU Efficiency {n_trees} trees {n_cores} cores',df,'name','cpu_efficiency',rotate_xticks=True)

    return df


def cpu_eff_subplots(data, num_trees, num_cores=[1, 4, 8, 12, 16]):
    dfs_plots = []
    # for num_tree in num_trees:
    #     df_1=df_cpu_eff_plot(data,1,n_trees=num_tree)
    #     df_4=df_cpu_eff_plot(data,4,n_trees=num_tree)
    #     df_8=df_cpu_eff_plot(data,8,n_trees=num_tree)
    #     df_12=df_cpu_eff_plot(data,12,n_trees=num_tree)
    #     df_16=df_cpu_eff_plot(data,16,n_trees=num_tree)
    #     dfs_plots.append(pd.DataFrame({
    #         'name': df_1['name'],
    #         'cpu_efficiency':[df_1['cpu_efficiency'].max(),df_4['cpu_efficiency'].max(),df_8['cpu_efficiency'].max(),df_12['cpu_efficiency'].max(),df_16['cpu_efficiency'].max()]
    #     }))

    for num_core in num_cores:
        df = df_cpu_eff_plot(data, num_core, num_trees)
        dfs_plots.append(df[["name", "n_cores_anomaly", "cpu_efficiency"]])

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    names = [
        "1 core",
        "4 cores",
        "8 cores",
        "12 cores",
        "16 cores",
    ]

    for i, (ax, name, df) in enumerate(zip(axes.flatten(), names, dfs_plots)):
        barplot_subplots(
            ax, name, df, "name", "cpu_efficiency", ylim_off=10, rotate_xticks=True
        )

    plt.tight_layout()
    plt.show()
