from __future__ import annotations

import os, sys, ctypes as c

from typing import ClassVar, Optional, List, Union

import numpy as np, numpy.typing as npt
from numba import njit, float64, int64, boolean
from numba.experimental import jitclass
from numba.typed import List
import ipdb

# get the current script dirpath
p = os.path.dirname(os.path.abspath(__file__))
sys.path.append(p)

from c_functions.c_signatures import Node, LeafData, dot_broadcast, copy_alloc


@njit(cache=True)
def make_rand_vector(df: int, dimensions: int) -> npt.NDArray[np.float64]:
    """
    Generate a random unitary vector in the unit ball with a maximum number of dimensions.
    This vector will be successively used in the generation of the splitting hyperplanes.

    Args:
        df: Degrees of freedom
        dimensions: number of dimensions of the feature space

    Returns:
        vec: Random unitary vector in the unit ball

    """
    if dimensions < df:
        raise ValueError("degree of freedom does not match with dataset dimensions")
    else:
        vec_ = np.random.normal(loc=0.0, scale=1.0, size=df).astype(np.float64)
        indexes = np.random.choice(np.arange(dimensions), df, replace=False)
        vec = np.zeros(dimensions, dtype=np.float64)
        vec[indexes] = vec_
        vec = vec / np.linalg.norm(vec)
    return vec


@njit(cache=True)
def c_factor(n: int) -> float:
    """
    Average path length of unsuccesful search in a binary search tree given n points.
    This is a constant factor that will be used as a normalization factor in the Anomaly Score calculation.

    Args:
        n: Number of data points for the BST.

    Returns:
        Average path length of unsuccesful search in a BST

    """
    if n <= 1:
        return 0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1.0) / (n * 1.0))


@njit(cache=True)
def get_leaf_ids(
    X: np.ndarray,
    child_left: np.ndarray,
    child_right: np.ndarray,
    normals: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    """
    Get the leaf node ids for each data point in the dataset.

    Args:
        X: Data points
        child_left: Left child node ids
        child_right: Right child node ids
        normals: Normal vectors of the splitting hyperplanes
        intercepts: Intercept values of the splitting hyperplanes

    Returns:
       Leaf node ids for each data point in the dataset.
    """
    res = []
    for x in X:
        node_id = 0
        while child_left[node_id] or child_right[node_id]:
            d = np.dot(np.ascontiguousarray(x), np.ascontiguousarray(normals[node_id]))
            node_id = (
                child_left[node_id]
                if d <= intercepts[node_id]
                else child_right[node_id]
            )
        res.append(int(node_id))
    return np.array(res)


tree_spec = [
    ("plus", boolean),
    ("locked_dims", int64),
    ("max_depth", int64),
    ("min_sample", int64),
    ("n", int64),
    ("d", int64),
    ("max_nodes", int64),
    ("child_left", int64[:]),
    ("child_right", int64[:]),
    ("normals", float64[:, :]),
    ("intercepts", float64[:]),
    ("corrected_depth", float64[:]),
    ("cumul_importance", float64[:, :]),
    ("eta", float64),
]


nodes: List[Node] = []


@jitclass(tree_spec)
class ExtendedTree:
    """
    Class that represents an Isolation Tree in the Extended Isolation Forest model.


    Attributes:
        plus (bool): Boolean flag to indicate if the model is a `EIF` or `EIF+`. Defaults to True (i.e. `EIF+`)
        locked_dims (int): Number of dimensions to be locked in the model. Defaults to 0
        max_depth (int): Maximum depth of the tree
        min_sample (int): Minimum number of samples in a node. Defaults to 1
        n (int): Number of samples in the dataset
        d (int): Number of dimensions in the dataset
        max_nodes (int): Maximum number of nodes in the tree. Defaults to 10000
        child_left (np.array): Array to store the left child nodes
        child_right (np.array): Array to store the right child nodes
        normals (np.array): Array to store the normal vectors of the splitting hyperplanes
        intercepts (np.array): Array to store the intercept values of the splitting hyperplanes
        node_size (np.array): Array to store the size of the nodes
        corrected_depth (np.array): Array to store the corrected depth of the nodes
        cumul_importance (np.array): Array to store the cumulative importances
        eta (float): Eta value for the model. Defaults to 1.5

    """

    def __init__(
        self,
        n: int,
        d: int,
        max_depth: int,
        locked_dims: int = 0,
        min_sample: int = 1,
        plus: bool = True,
        max_nodes: int = 10000,
        eta: float = 1.5,
    ):

        self.plus = plus
        self.locked_dims = locked_dims
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.n = n
        self.d = d
        self.max_nodes = max_nodes
        self.eta = eta

        self.child_left = np.zeros(max_nodes, dtype=np.int64)
        self.child_right = np.zeros(max_nodes, dtype=np.int64)
        self.normals = np.zeros((max_nodes, d), dtype=np.float64)
        self.intercepts = np.zeros(max_nodes, dtype=np.float64)
        self.corrected_depth = np.zeros(max_nodes, dtype=np.float64)
        self.cumul_importance = np.zeros((max_nodes, d), dtype=np.float64)
        self.cumul_normals = np.zeros((max_nodes, d), dtype=np.float64)

        # self.nodes = np.array([], dtype=c.POINTER(Node))

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model to the dataset.

        Args:
            X: Input dataset

        Returns:
            The method fits the model and does not return any value.
        """
        self.extend_tree(node_id=0, X=X, depth=0)
        self.corrected_depth = self.corrected_depth / c_factor(len(X))

    def extend_tree(self, node_id: int, X: npt.NDArray, depth: int) -> None:
        """
        Extend the tree to the given node.

        Args:
            node_id: Node id
            X: Input dataset
            depth: Depth of the node

        Returns:
            The method extends the tree and does not return any value.
        """

        if self.plus:
            make_random_intercept = lambda dist: np.random.normal(
                np.mean(dist), np.std(dist) * self.eta
            )
        else:
            make_random_intercept = lambda dist: np.random.uniform(
                np.min(dist), np.max(dist)
            )

        nodes_p_lst = []

        X = X.astype(np.float64)

        self.ids = np.arange(len(X), dtype=np.uint32)

        def create_split(
            node: Node,
            subset_ids: np.ndarray,
            depth: int,
            cumul_importance: np.ndarray,
            cumul_normals: np.ndarray,
        ) -> None:

            node_size = subset_ids.shape[0]

            if node_size <= self.min_sample or depth >= self.max_depth:
                # reached a leaf node
                self.corrected_depth[node.id] = c_factor(node_size) + depth + 1
                self.cumul_normals[node.id] = cumul_normals
                self.cumul_importance[node.id] = cumul_importance
                node.is_leaf = True  # C
                node.leaf_data.cumul_normals = self.corrected_depth[node.id]  # C
                node.leaf_data.cumul_importance = cumul_normals  # C
                node.leaf_data.corrected_depth = cumul_importance  # C

                self.ids[subset_ids] = node.id
                return
            else:
                node.is_leaf = False  # C
                node.leaf_data = None  # C

            normal = make_rand_vector(self.d - self.locked_dims, self.d)
            self.normals[node.id] = normal

            # DOES NOT WORK:
            # node.normal = (c.c_double * normal.shape[0])()
            # c.memmove(node.normal, normal.ctypes.data, normal.shape[0])

            # node.normal = normal.ctypes.data_as(c.POINTER(c.c_double))  # DOES NOT WORK

            # node.normal = self.normals[node.id].ctypes.data_as(c.POINTER(c.c_double))  # WORKS

            node.normal = copy_alloc(normal)

            dist = np.dot(
                np.ascontiguousarray(X[subset_ids]),
                np.ctypeslib.as_array(node.normal, shape=(X.shape[1],)).astype(
                    np.float64
                ),
            )

            # dist = dot_broadcast(np.ascontiguousarray(X[subset_ids]), node.normal)

            # if not np.array_equal(dist,dist1):
            #     ipdb.set_trace()

            self.intercepts[node.id] = make_random_intercept(dist)
            node.intercept = self.intercepts[node.id].astype(c.c_double)

            mask = dist <= self.intercepts[node.id]

            partial_importance = np.abs(normal)
            cumul_normals += partial_importance
            partial_importance *= node_size

            # LEFT
            left_node = Node()
            node.left_child_id = left_node.id = self.child_left[node.id] = len(
                nodes_p_lst
            )
            nodes_p_lst.append(c.pointer(left_node))
            if len(nodes_p_lst) >= self.max_nodes:
                raise ValueError("Max number of nodes reached")
            l_cumul_importance = cumul_importance + partial_importance / (
                len(subset_ids[mask]) + 1
            )

            # RIGHT
            right_node = Node()
            node.right_child_id = right_node.id = self.child_right[node.id] = len(
                nodes_p_lst
            )
            nodes_p_lst.append(c.pointer(right_node))
            if len(nodes_p_lst) >= self.max_nodes:
                raise ValueError("Max number of nodes reached")
            r_cumul_importance = cumul_importance + partial_importance / (
                len(subset_ids[~mask]) + 1
            )

            depth += 1

            create_split(
                right_node,
                subset_ids[~mask],
                depth,
                r_cumul_importance,
                cumul_normals.copy(),
            )
            create_split(
                left_node,
                subset_ids[mask],
                depth,
                l_cumul_importance,
                cumul_normals,
            )

        subset_ids = np.arange(len(X), dtype=np.uint32)
        root_node = Node(id=node_id)
        nodes_p_lst.append(c.pointer(root_node))
        create_split(
            root_node, subset_ids, depth, np.zeros(X.shape[1]), np.zeros(X.shape[1])
        )

        self.nodes = (c.POINTER(Node) * len(nodes_p_lst))(*nodes_p_lst)

    def leaf_ids(self, X: np.ndarray) -> np.ndarray:
        """
        Get the leaf node ids for each data point in the dataset.

        This is a stub method of `get_leaf_ids`.

        Args:
            X: Input dataset

        Returns:
           Leaf node ids for each data point in the dataset.
        """
        leaf_ids = np.zeros(X.shape[0], dtype=c.c_uint)
        for i, x in enumerate(X):
            node = self.nodes[0].contents
            while not node.is_leaf:
                d = np.dot(
                    np.ascontiguousarray(x),
                    np.ctypeslib.as_array(node.normal, shape=(x.shape)).astype(
                        np.float64
                    ),
                )
                node_id = (
                    node.left_child_id if d <= node.intercept else node.right_child_id
                )
                node = self.nodes[node_id].contents

            leaf_ids[i] = node.id

        return leaf_ids

    def predict(self, ids: np.ndarray) -> np.ndarray:
        """
        Predict the anomaly score for each data point in the dataset.

        Args:
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            Anomaly score for each data point in the dataset.
        """
        return self.corrected_depth[ids]

    def importances(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the importances of the features for the given leaf node ids.

        Args:
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            Importances of the features for the given leaf node ids and the normal vectors.
        """

        return self.cumul_importance[ids], self.cumul_normals[ids]


class ExtendedIsolationForest:
    """
    Class that represents the Extended Isolation Forest model.

    Attributes:
        n_estimators (int): Number of trees in the model. Defaults to 400
        max_samples (int): Maximum number of samples in a node. Defaults to 256
        max_depth (int): Maximum depth of the trees. Defaults to "auto"
        plus (bool): Boolean flag to indicate if the model is a `EIF` or `EIF+`.
        name (str): Name of the model
        ids (np.array): Leaf node ids for each data point in the dataset. Defaults to None
        X (np.array): Input dataset. Defaults to None
        eta (float): Eta value for the model. Defaults to 1.5
        avg_number_of_nodes (int): Average number of nodes in the trees

    """

    def __init__(
        self,
        plus: bool,
        n_estimators: int = 400,
        max_depth: Union[str, int] = "auto",
        max_samples: Union[str, int] = "auto",
        eta: float = 1.5,
    ):
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.max_depth = max_depth
        self.plus = plus
        self.name = "EIF" + "+" * int(plus)
        self.ids = None
        self.X = None
        self.eta = eta

    @property
    def avg_number_of_nodes(self):
        return np.mean([len(T.nodes) for T in self.trees])

    def fit(self, X: np.ndarray, locked_dims: int | None = None) -> None:
        """
        Fit the model to the dataset.

        Args:
            X: Input dataset
            locked_dims: Number of dimensions to be locked in the model. Defaults to None

        Returns:
            The method fits the model and does not return any value.
        """

        self.ids = None
        if not locked_dims:
            locked_dims = 0

        if self.max_depth == "auto":
            self.max_depth = int(np.ceil(np.log2(self.max_samples)))
        subsample_size = np.min((self.max_samples, len(X)))
        self.trees = [
            ExtendedTree(
                subsample_size,
                X.shape[1],
                self.max_depth,
                locked_dims=locked_dims,
                plus=self.plus,
                eta=self.eta,
            )
            for _ in range(self.n_estimators)
        ]
        for T in self.trees:
            T.fit(X[np.random.randint(len(X), size=subsample_size)])

    def compute_ids(self, X: np.ndarray) -> None:
        """
        Compute the leaf node ids for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            The method computes the leaf node ids and does not return any value.
        """
        # if the results
        if (
            self.ids is None
            or self.X.shape != X.shape
            or np.array_equal(self.X, X) is False
        ):
            self.X = X
            self.ids = np.array([tree.leaf_ids(X) for tree in self.trees])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the anomaly score for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            Anomaly score for each data point in the dataset.
        """
        self.compute_ids(X)
        predictions = [tree.predict(self.ids[i]) for i, tree in enumerate(self.trees)]
        return np.power(2, -np.mean(predictions, axis=0))

    def _predict(self, X: np.ndarray, p: float) -> np.ndarray:
        """
        Predict the class of each data point (i.e. inlier or outlier) based on the anomaly score.

        Args:
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score)

        Returns:
           Class labels (i.e. 0 for inliers and 1 for outliers)
        """
        anomaly_score = self.predict(X)
        y_hat = anomaly_score > np.percentile(
            anomaly_score, 100 * (1 - p), method="inverted_cdf"
        )
        return y_hat

    def _importances(
        self, X: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the importances of the features for the given leaf node ids.

        Args:
            X: Input dataset
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            Importances of the features for the given leaf node ids and the normal vectors.

        """
        importances = np.zeros(X.shape)
        normals = np.zeros(X.shape)
        for i, tree in enumerate(self.trees):
            importance, normal = tree.importances(ids[i])
            importances += importance
            normals += normal
        return importances / self.n_estimators, normals / self.n_estimators

    def global_importances(self, X: np.ndarray, p: float = 0.1) -> np.ndarray:
        """
        Compute the global importances of the features for the dataset.

        Args:
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score). Defaults to 0.1

        Returns:
            Global importances of the features for the dataset.
        """

        self.compute_ids(X)
        y_hat = self._predict(X, p)
        importances, normals = self._importances(X, self.ids)
        outliers_importances, outliers_normals = np.sum(
            importances[y_hat], axis=0
        ), np.sum(normals[y_hat], axis=0)
        inliers_importances, inliers_normals = np.sum(
            importances[~y_hat], axis=0
        ), np.sum(normals[~y_hat], axis=0)
        return (outliers_importances / outliers_normals) / (
            inliers_importances / inliers_normals
        )

    def local_importances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the local importances of the features for the dataset.

        Args:
            X: Input dataset

        Returns:
           Local importances of the features for the dataset.
        """

        self.compute_ids(X)
        importances, normals = self._importances(X, self.ids)
        return importances / normals


class IsolationForest(ExtendedIsolationForest):
    """
    Class that represents the Isolation Forest model.

    This is a subclass of `ExtendedIsolationForest` with the `plus` attribute set to False and the
    `locked_dims` attribute set to the number of dimensions minus one.

    Attributes:
        n_estimators (int): Number of trees in the model. Defaults to 400
        max_depth (Union[str,int]): Maximum depth of the trees. Defaults to "auto"
        max_samples (Union[str,int]): Maximum number of samples in a node. Defaults to "auto"

    """

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: Union[str, int] = "auto",
        max_samples: Union[str, int] = "auto",
    ) -> None:
        super().__init__(
            plus=False,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,
        )
        self.name = "IF"

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the model to the dataset.

        Args:
            X: Input dataset

        Returns:
            The method fits the model and does not return any value.
        """

        return super().fit(X, locked_dims=X.shape[1] - 1)

    def decision_function_single_tree(
        self, tree_idx: int, X: np.ndarray, p: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the anomaly score for each data point in the dataset using a single tree.

        Args:
            tree_idx: Index of the tree
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score). Defaults to 0.1

        Returns:
            Anomaly score for each data point in the dataset and the predicted class for each data point in the dataset.
        """

        self.compute_ids(X)
        pred = self.trees[tree_idx].predict(X, self.ids[tree_idx])[0]
        score = np.power(2, -pred)
        y_hat = np.array(
            score > sorted(score, reverse=True)[int(p * len(score))], dtype=int
        )
        return score, y_hat
