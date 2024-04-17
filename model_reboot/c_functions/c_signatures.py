import os, ctypes as c
import numpy as np, numpy.typing as npt
from glob import glob


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
    ]


p = os.path.dirname(os.path.abspath(__file__))
