import ctypes
import timeit
import pickle as pkl
import numpy as np
from c_signatures import c_compute_paths, Node


def nodes_to_c_array(nodes, dest_arr, num_features: int):
    """
    Convert node dictionary to a ctypes array of Node structure instances.
    Nodes example:
    ```
    nodes = {
      0:{"point":0.11123, "normal":[12,3,3,5], "numerosity":45},
      1:{"point":0.11123, "normal":[12,3,3,5], "numerosity":45}},
      2:{....},
    }
    ```
    Parameters:
    - nodes: dictionary of dictionaries. The keys are the node ids and the values
        are dictionaries containing the node information.
    - dest_arr: a ctypes array of Node structure instances.
    """
    for i, node in enumerate(nodes):
        dest_arr[i].numerosity = ctypes.c_int(node["numerosity"])
        dest_arr[i].is_leaf = ctypes.c_bool(node["point"] is None)
        # allocate the memory for the normal array
        dest_arr[i].normal = (ctypes.c_double * num_features)()
        if node["point"] is None:
            continue
        else:
            dest_arr[i].point = ctypes.c_double(node["point"])
            dest_arr[i].normal = (ctypes.c_double * num_features)(*node["normal"])


def convert2c_paths_args(X, nodes, left_son, right_son, c_paths):

    c_X = np.array(X, dtype=np.float64)
    c_X = c_X.flatten()

    # --- Create the data structures for the C function ---
    nodes_len = int(len(nodes))
    X_rows, X_cols = int(X.shape[0]), int(X.shape[1])

    # convert the nodes dictionary to a ctypes array of Node structure instances
    NodeArray = Node * len(nodes)
    c_node_arr = NodeArray()
    nodes_to_c_array(nodes, c_node_arr, X_cols)

    # convert the lists in numpy arrays
    c_left_son = np.array(left_son, dtype=np.int32)
    c_right_son = np.array(right_son, dtype=np.int32)

    c_X_rows, c_X_cols = ctypes.c_int(X_rows), ctypes.c_int(X_cols)

    return (
        c_X,
        c_node_arr,
        c_left_son,
        c_right_son,
        c_X_rows,
        c_X_cols,
        c_paths,
    )


def c_compute_paths_wrapper(X, nodes, left_son, right_son):
    paths = np.zeros(X.shape[0], dtype=np.int32)
    args = convert2c_paths_args(X, nodes, left_son, right_son, paths)
    return c_compute_paths_wrapper_clean(args)


def c_compute_paths_wrapper_clean(args):
    c_compute_paths(*args)
    return args[-1]


def py_compute_paths_wrapper(
    X: np.ndarray, nodes, left_son: np.ndarray, right_son: np.ndarray
):
    def compute_paths(X: np.ndarray):
        """
        Compute the path followed by sample from the root to the leaf where it is contained.
        The result of this function is used for the Anomaly Score computation.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: np array
                Input dataset
        Returns
        ----------
        paths: List
                List of nodes encountered by a sample in its path towards the leaf in which it is contained.
        """

        def path(x):
            k = 1
            id = 0
            while True:
                s = nodes[id]["point"]
                if s is None:
                    break
                n = nodes[id]["normal"]
                if x.dot(n) - s > 0:
                    id = left_son[id]
                else:
                    id = right_son[id]
                k += 1
            return k

        return np.apply_along_axis(path, 1, X)

    return compute_paths(X)


# --- Load example data ---
moody = "/home/frabors/code-files/phd-code/HPC-Project-AD/src/make_importance_c/nodes_moody.pkl"
wine = "/home/frabors/code-files/phd-code/HPC-Project-AD/src/make_importance_c/nodes_wine.pkl"
data = None
with open(wine, "rb") as f:
    data = pkl.load(f)

nodes = np.array(list(data["nodes"].values()))
left_son = data["left_son"]
right_son = data["right_son"]
X: np.ndarray = data["X"]


# --- Compare the Python and C functions output ---
print("... Equality check (CTRL+C to skip) ...")
try:
    c_paths = c_compute_paths_wrapper(X, nodes, left_son, right_son)
    py_paths = py_compute_paths_wrapper(X, nodes, left_son, right_son)

    print("Difference in paths:", np.abs(py_paths - c_paths).mean())
except KeyboardInterrupt:
    pass

print("\n... Timing tests ...")
# --- Compare the Python and C functions execution times ---
num_runs = 100
python_time = timeit.timeit(
    lambda: py_compute_paths_wrapper(X, nodes, left_son, right_son),
    number=num_runs,
)

c_time = timeit.timeit(
    lambda: c_compute_paths_wrapper(X, nodes, left_son, right_son),
    number=num_runs,
)
paths = np.zeros(X.shape[0], dtype=np.int32)
args = convert2c_paths_args(X, nodes, left_son, right_son, paths)
c_time_clean = timeit.timeit(
    lambda: c_compute_paths_wrapper_clean(args),
    number=num_runs,
)

print("num runs", num_runs)
print(f"Python time: {python_time/num_runs}")
print(f"C time (with input data conversion overhead): {c_time/num_runs}")
print(f"C time (no input data conversion overhead): {c_time_clean/num_runs}")

print(
    "C speedup (with input data conversion overhead):", 100 * python_time / c_time, r"%"
)
print(
    "C speedup (no input data conversion overhead):",
    100 * python_time / c_time_clean,
    r"%",
)
