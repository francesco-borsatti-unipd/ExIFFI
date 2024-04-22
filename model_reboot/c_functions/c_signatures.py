import os, ctypes as c
import numpy as np, numpy.typing as npt


class LeafData(c.Structure):
    """
    C structure that represents the data in a Leaf Node.
    """

    _fields_ = [
        ("cumul_normals", c.POINTER(c.c_double)),
        ("cumul_importance", c.POINTER(c.c_double)),
        ("corrected_depth", c.c_double),
    ]


class Node(c.Structure):
    """
    C structure that represents a Node in the tree.
    """

    _fields_ = [
        ("intercept", c.c_double),
        ("normal", c.POINTER(c.c_double)),
        ("left_child_id", c.c_uint),
        ("right_child_id", c.c_uint),
        ("leaf_data", c.POINTER(LeafData)),
        ("id", c.c_uint),
        ("is_leaf", c.c_bool),
    ]


p = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(p, "functions_lib.so")

if os.environ.get("COMPILE_C_EXIFFI") == "1":
    src_path = os.path.join(p, "functions_lib.c")
    os.system(
        f"gcc -Wall -pedantic -shared -fPIC -O2 -fopenmp -lm -o '{lib_path}' '{src_path}'"
    )

lib = c.CDLL(lib_path)

# -- C DOT BROADCAST ----------------------------------------
c_dot_broadcast = lib.dot_broadcast
c_dot_broadcast.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    c.POINTER(c.c_double),
    c.c_uint,
    c.c_uint,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
]
c_dot_broadcast.restype = c.POINTER(c.c_double)


def dot_broadcast(a: npt.NDArray[np.float64], b) -> npt.NDArray[np.float64]:
    """
    Dot product between two arrays.
    """
    res = np.zeros(a.shape[0], dtype=np.float64)
    c_dot_broadcast(a.flatten(), b, a.shape[0], a.shape[1], res)
    return res


# -- C COPY ALLOC -------------------------------------------
c_copy_alloc = lib.copy_alloc
c_copy_alloc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    c.c_uint,
]
c_copy_alloc.restype = c.POINTER(c.c_double)


def copy_alloc(a: npt.NDArray[np.float64]):
    return c_copy_alloc(a, a.shape[0])


# -- C GET LEAF IDS -----------------------------------------
c_get_leaf_ids = lib.get_leaf_ids
c_get_leaf_ids.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    c.c_uint,
    c.c_uint,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    c.POINTER(c.POINTER(Node)),
    c.c_uint,
]
c_get_leaf_ids.restype = None


def get_leaf_ids(
    X: npt.NDArray[np.float64], leaf_ids: npt.NDArray[np.int32], nodes, n_nodes: int
):
    """
    Parameters
    - nodes: array of pointers to Node structures
    """
    c_get_leaf_ids(X.flatten(), X.shape[0], X.shape[1], leaf_ids, nodes, n_nodes)


# -- C C_FACTOR ---------------------------------------------
c_c_factor = lib.c_factor
c_c_factor.argtypes = [c.c_int]
c_c_factor.restype = c.c_double


def c_factor(n: int) -> float:
    return c_c_factor(n)


# -- C SAVE_LEAF_DATA ---------------------------------------
c_save_leaf_data = lib.save_leaf_data
c_save_leaf_data.argtypes = [
    c.POINTER(Node),
    c.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    c.c_uint,
]


def save_leaf_data(node, corrected_depth, cumul_normals, cumul_importance):
    c_save_leaf_data(
        node, corrected_depth, cumul_normals, cumul_importance, len(cumul_normals)
    )


# -- C GET_CORRECTED_DEPTH ----------------------------------
c_get_corrected_depths = lib.get_corrected_depths
c_get_corrected_depths.argtypes = [
    c.POINTER(c.POINTER(Node)),
    c.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    c.c_int,
]
c_get_corrected_depths.restype = None


def get_corrected_depths(nodes, ids, num_nodes):
    """
    Get the corrected depth of the nodes.
    """
    corrected_depth = np.zeros(len(ids), dtype=np.float64)
    c_get_corrected_depths(nodes, num_nodes, corrected_depth, ids, len(ids))
    return corrected_depth


# -- C SAVE_TRAIN_DATA --------------------------------------
# memory expensive technique
c_save_train_data = lib.save_train_data
c_save_train_data.argtypes = [
    c.POINTER(c.POINTER(Node)),  # **nodes
    c.c_int,  # num_nodes
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    ),  # *corrected_depths
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    ),  # **cumul_importance
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    ),  # **cumul_normals
    c.c_int,  # vec_len
]
c_save_train_data.restype = None


def save_train_data(nodes, d):
    num_nodes = len(nodes)

    corrected_depths = np.zeros(num_nodes, dtype=np.float64)
    cumul_importances = np.zeros((num_nodes * d), dtype=np.float64)
    cumul_normals = np.zeros((num_nodes * d), dtype=np.float64)

    c_save_train_data(
        nodes,
        num_nodes,
        corrected_depths,
        cumul_importances,
        cumul_normals,
        d,
    )

    return (
        corrected_depths,
        cumul_importances.reshape((num_nodes, d)),
        cumul_normals.reshape((num_nodes, d)),
    )
