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
    Dot product between two arrays, where the second array is broadcasted to the first.

    Parameters
        a: array of shape (n_samples, n_features)
        b: array of shape (n_features)

    Returns
        res: array of shape (n_samples) where each element is the dot product of the
            corresponding row of a with b.
    """
    res = np.zeros(a.shape[0], dtype=np.float64)
    c_dot_broadcast(a.flatten(), b, a.shape[0], a.shape[1], res)
    return res


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
        X: array of shape (n_samples, n_features) where we want to find where each sample
            ends up in the tree, i.e. the leaf node it belongs to.
        leaf_ids: array of integers where the leaf node id of each sample will be stored
        nodes: array of pointers to Node structures
        n_nodes: number of nodes in the tree

    Returns
        None, the leaf_ids array is modified in place with the leaf node id of each sample.
    """
    c_get_leaf_ids(X.flatten(), X.shape[0], X.shape[1], leaf_ids, nodes, n_nodes)


# -- C C_FACTOR ---------------------------------------------
c_c_factor = lib.c_factor
c_c_factor.argtypes = [c.c_int]
c_c_factor.restype = c.c_double


def c_factor(n: int) -> float:
    """
    Compute the correction factor given the number of samples in a node.

    Parameters
        n: number of samples in the node

    Returns
        c_factor: correction factor
    """
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
    """
    Given a leaf node, allocate memory and save the data in the leaf node.

    Parameters
        node: pointer to the leaf node
        corrected_depth: corrected depth of the node
        cumul_normals: array of shape (n_features) with the sum of the normal vectors of the
            normals of each split along the path to this leaf node
        cumul_importance: array of shape (n_features) with the cumulative sum of the importances
            along the path of to this leaf node

    Returns
        None, the data is saved in the leaf node.
    """
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
    Read the tree nodes and get the corrected depth of the nodes with the given ids.

    Parameters
        nodes: array of pointers to Node structures
        ids: array of integers with the ids of the nodes we want to get the corrected depth of
        num_nodes: number of nodes in the tree
    
    Returns
        corrected_depth: array of shape (len(ids)) with the corrected depth of the nodes with
            the given ids
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
    """
    Create arrays to store the corrected depths, cumulative importances and cumulative normals
    and read the respective tree leaf nodes to save the data in the arrays.

    Parameters
        nodes: array of pointers to Node structures
        d: number of features

    Returns
        corrected_depths: array of shape (num_nodes) with the corrected depth of each node
        cumul_importances: array of shape (num_nodes, d) with the cumulative sum of the importances
            along the path of each node
        cumul_normals: array of shape (num_nodes, d) with the sum of the normal vectors of the
            normals of each split along the path of each node
    """
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


# -- C ALLOC_CHILD_NODES ------------------------------------
c_alloc_child_nodes = lib.alloc_child_nodes
c_alloc_child_nodes.argtypes = [
    c.POINTER(c.POINTER(Node)),
    c.c_int,
    c.c_int,
    c.POINTER(Node),
]
c_alloc_child_nodes.restype = None


def alloc_child_nodes(nodes, max_nodes, num_nodes, current_node):
    """
    If the max number of nodes is not reached, allocate memory for the left and right child nodes
    of the current node. Also, update the node ids of the relevant nodes.

    Parameters
        nodes: array of pointers to Node structures
        max_nodes: maximum number of nodes in the tree
        num_nodes: number of nodes in the tree
        current_node: pointer to the current node

    Returns
        None, the memory is allocated for the left and right child nodes of the current node
        and the node ids are updated.
    """
    c_alloc_child_nodes(nodes, max_nodes, num_nodes, current_node)


# -- C SAVE_NORMAL_VECTOR -----------------------------------
c_save_normal_vector = lib.save_normal_vector
c_save_normal_vector.argtypes = [
    c.POINTER(Node),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    c.c_int,
]
c_save_normal_vector.restype = None


def save_normal_vector(node, normal):
    """
    Allocate memory for the normal vector of the node and copy the normal vector to the node.

    Parameters
        node: pointer to the node where the normal vector will be saved
        normal: array of shape (d) with the normal vector of the node
    """
    c_save_normal_vector(node, normal, len(normal))


# -- C UPDATE_IMPORTANCES_AND_NORMALS -----------------------
c_update_importances_and_normals = lib.update_importances_and_normals
c_update_importances_and_normals.argtypes = [
    c.POINTER(Node),
    c.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.bool_, ndim=1, flags="C_CONTIGUOUS"),
    c.c_int,
]
c_update_importances_and_normals.restype = None


def update_importances_and_normals(
    node,
    d,
    cumul_normal,
    cumul_importance,
    mask,
):
    """
    Compute the cumulative sum of the importances and the sum of the normal vectors of the
    normals for the current node split (left and right child nodes).

    Parameters
        node: pointer to the node
        d: number of features
        cumul_normal: array of shape (d) with the sum of the normal vectors of the normals of
            each split along the path up to this node
        cumul_importance: array of shape (d) with the cumulative sum of the importances along
            the path up to this node
        mask: array of shape (d) with the boolean mask of samples that will be split to the left
            and right child nodes.
    """
    l_cumul_importance = np.zeros(d)
    r_cumul_importance = np.zeros(d)
    c_update_importances_and_normals(
        node,
        d,
        cumul_normal,
        cumul_importance,
        l_cumul_importance,
        r_cumul_importance,
        mask,
        mask.shape[0],
    )
    return l_cumul_importance, r_cumul_importance
