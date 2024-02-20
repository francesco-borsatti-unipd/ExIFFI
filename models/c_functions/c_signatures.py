import os
import ctypes
import numpy as np
from glob import glob
import numpy.typing as npt


class Node(ctypes.Structure):
    """
    C structure that represents a Node in the tree.
    """

    _fields_ = [
        ("point", ctypes.c_double),
        ("normal", ctypes.POINTER(ctypes.c_double)),
        ("numerosity", ctypes.c_int),
        ("is_leaf", ctypes.c_bool),
    ]


p = os.path.dirname(os.path.abspath(__file__))

# --- COMPUTE PATHS FUNCTION ---
lib_path = glob(os.path.join(p, "c_compute_paths.so"))
assert lib_path, f"c_compute_paths.so not found in path {p}"
lib = ctypes.CDLL(lib_path[0])
c_compute_paths_raw = lib.compute_paths
c_compute_paths_raw.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # dataset
    ctypes.POINTER(Node),  # nodes array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # left_son
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # right_son
    ctypes.c_int,  # number of rows
    ctypes.c_int,  # number of columns
    np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"
    ),  # computed paths destination array
]


def c_compute_paths(
    X: npt.NDArray[np.float64],
    nodes,
    left_son: npt.NDArray[np.int32],
    right_son: npt.NDArray[np.int32],
    X_rows: ctypes.c_int,
    X_cols: ctypes.c_int,
    paths: npt.NDArray[np.int32],
):
    """
    Compute paths using C implementation.

    Parameters:
        X (np.ndarray): Input flattened dataset array .
        nodes: Nodes information.
        left_son (np.ndarray): Left son array.
        right_son (np.ndarray): Right son array.
        paths (np.ndarray): The array where the computed paths will be saved to.

    Returns:
        None. The computed paths are in the `paths` array.
    """
    c_compute_paths_raw(X, nodes, left_son, right_son, X_rows, X_cols, paths)


# --- ANOMALY SCORE FUNCTION ---
lib_path = glob(os.path.join(p, "c_anomaly_score.so"))
assert lib_path, f"c_anomaly_score.so not found in path {p}"
lib = ctypes.CDLL(lib_path[0])
c_anomaly_score_raw = lib.anomaly_score
c_anomaly_score_raw.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # dataset
    # this time, Nodes is an array of arrays of Node
    ctypes.POINTER(ctypes.POINTER(Node)),  # nodes array
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # forest_left_son
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # forest_right_son
    ctypes.c_int,  # number of trees
    ctypes.c_int,  # number of rows
    ctypes.c_int,  # number of cols
    np.ctypeslib.ndpointer(
        dtype=np.double, ndim=1, flags="C_CONTIGUOUS"
    ),  # result array shape (n_samples,)
]


def c_anomaly_score(
    X: npt.NDArray[np.float64],
    forest_nodes,
    forest_left_son,
    forest_right_son,
    num_trees: ctypes.c_int,
    X_rows: ctypes.c_int,
    X_cols: ctypes.c_int,
    anomaly_scores: npt.NDArray[np.float64],
):
    """
    Compute anomaly score using C implementation.

    Parameters:
        X (np.ndarray): Input flattened dataset array .
        forest_nodes: array of array of Node struct.
        forest_left_son: array of left_son arrays.
        forest_right_son: array of right_son arrays.
        n_trees (int): Number of trees in the forest.
        X_rows (int): Number of rows in the X dataset.
        X_cols (int): Number of columns in the X dataset.
        anomaly_scores (np.ndarray): The array where the computed anomaly scores will be saved to.

    Returns:
        None. The computed anomaly scores are in the `paths` array.
    """
    c_anomaly_score_raw(
        X,
        forest_nodes,
        forest_left_son,
        forest_right_son,
        num_trees,
        X_rows,
        X_cols,
        anomaly_scores,
    )


# --- MAKE IMPORTANCE FUNCTION ---
lib_path = glob(os.path.join(p, "c_make_importance.so"))
assert lib_path, f"c_make_importance.so not found in path {p}"
lib = ctypes.CDLL(lib_path[0])
c_make_importance_raw = lib.make_importance

# define the C function signature
c_make_importance_raw.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # dataset
    ctypes.POINTER(Node),  # nodes array
    ctypes.c_bool,  # depth_based
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # left_son
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # right_son
    ctypes.c_int,  # number of rows
    ctypes.c_int,  # number of columns
    np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"
    ),  # paths DESTINATION ARRAY
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    ),  # importances DESTINATION ARRAY
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    ),  # normal vectors DESTINATION ARRAY
)


def c_make_importance(
    X: npt.NDArray[np.float64],
    nodes,
    depth_based: ctypes.c_bool,
    left_son: npt.NDArray[np.int32],
    right_son: npt.NDArray[np.int32],
    X_rows: ctypes.c_int,
    X_cols: ctypes.c_int,
    paths: npt.NDArray[np.int32],
    importances: npt.NDArray[np.float64],
    normal_vectors: npt.NDArray[np.float64],
):
    """
    Compute the Importance Scores for each node along the Isolation Tree.

    Parameters
        X (np.array):  Input dataset
        depth_based (bool): decide weather to used the Depth Based or the Not Depth Based Importance computation function.

    Returns
        Importances_arr: np.array of the Importances values for all the nodes in the Isolation Tree.
        Normal_vectors_list: np.array of the normal vectors for all the nodes in the Isolation Tree.
    """

    c_make_importance_raw(
        X,
        nodes,
        depth_based,
        left_son,
        right_son,
        X_rows,
        X_cols,
        paths,
        importances,
        normal_vectors,
    )
