import ctypes
import numpy as np


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


# compute paths function
lib = ctypes.CDLL("./c_compute_paths.so")
c_compute_paths = lib.compute_paths
c_compute_paths.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # dataset
    ctypes.POINTER(Node),  # nodes array
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # left_son
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # right_son
    ctypes.c_int,  # number of nodes (also number of left and right sons)
    ctypes.c_int,  # number of rows
    ctypes.c_int,  # number of columns
    np.ctypeslib.ndpointer(
        dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"
    ),  # computed paths destination array
]
c_compute_paths.restype = ctypes.c_void_p


# make importance function
lib = ctypes.CDLL("./c_make_importance.so")
c_importance = lib.importance_worker

# define the C function signature
c_importance.argtypes = (
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # dataset
    ctypes.POINTER(Node),  # nodes array
    ctypes.c_bool,  # depth_based
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # left_son
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # right_son
    ctypes.c_int,  # number of nodes (also number of left and right sons)
    ctypes.c_int,  # number of rows
    ctypes.c_int,  # number of columns
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    ),  # importances DESTINATION ARRAY
    np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"
    ),  # normal vectors DESTINATION ARRAY
)
