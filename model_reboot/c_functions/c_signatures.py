import os, ctypes as c
import numpy as np, numpy.typing as npt


class LeafData(c.Structure):
    """
    C structure that represents the data in a Leaf Node.
    """

    _fields_ = [
        ("cumul_normals", c.c_double),
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
src_path = os.path.join(p, "functions_lib.c")
lib_path = os.path.join(p, "functions_lib.so")

os.system(f"gcc -Wall -pedantic -shared -fPIC -o {lib_path} {src_path}")

lib = c.CDLL(lib_path)

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


c_copy_alloc = lib.copy_alloc
c_copy_alloc.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    c.c_uint,
]
c_copy_alloc.restype = c.POINTER(c.c_double)


def copy_alloc(a: npt.NDArray[np.float64]):
    return c_copy_alloc(a, a.shape[0])
