"""
Extended Isolation Forest model 
"""

import sys
import os
from typing import Optional, Any

p = os.path.dirname(os.path.abspath(__file__))
p = os.path.dirname(p)
sys.path.append(p)

import ctypes
from functools import partial
from multiprocessing import Pool
import numpy.typing as npt

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

from utils.utils import make_rand_vector, c_factor
from models.c_functions import Node, c_compute_paths, c_anomaly_score


class ExtendedTree:
    def __init__(self, dims, min_sample, max_depth, plus):
        """
        Implementation of Isolation Trees for the EIF/EIF_plus models
        --------------------------------------------------------------------------------

        Parameters
        ----------
        dims: int
                Number of degrees of freedom used in the separating hyperplanes
        min_sample: int
                Minimum number of samples in a node where isolation is achieved
        max_depth: int
                Maximum depth at which a sample can be considered as isolated
        plus: int
                This parameter is used to distinguish the EIF and the EIF_plus models. If plus=0 then the EIF model
                will be used, if plus=1 than the EIF_plus model will be considered.
        """
        self.dims = dims
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.depth = 0
        self.right_son = [0]
        self.left_son = [0]
        self.nodes = {}
        self.plus = plus

    def make_tree(self, X, id, depth):
        """
        Create an Isolation Tree using the separating hyperplanes
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        id: int
                Node index
        depth: int
                Depth value

        N.B The id and depth input parameters are needed because this is a recursive algorithm. At the first call id and depth
        will be equal to 0.

        """
        if X.shape[0] <= self.min_sample:
            self.nodes[id] = {"point": None, "normal": None, "numerosity": len(X)}
        elif depth >= self.max_depth:
            self.nodes[id] = {"point": None, "normal": None, "numerosity": len(X)}
        else:
            n = make_rand_vector(self.dims, X.shape[1])

            val = X.dot(n)
            s = (
                np.random.normal(np.mean(val), np.std(val) * 2)
                if np.random.random() < self.plus
                else np.random.uniform(np.min(val), np.max(val))
            )
            lefts = val > s

            self.nodes[id] = {"point": s, "normal": n, "numerosity": len(X)}

            idsx = len(self.nodes)
            self.left_son[id] = int(idsx)
            self.right_son.append(0)
            self.left_son.append(0)
            self.make_tree(X[lefts], idsx, depth + 1)

            iddx = len(self.nodes)
            self.right_son.append(0)
            self.left_son.append(0)
            self.right_son[id] = int(iddx)
            self.make_tree(X[~lefts], iddx, depth + 1)

    def compute_paths2(self, X, id, true_vec=None):
        """
        Compute the path followed by sample from the root to the leaf where it is contained. The result of this
        function is used for the Anomaly Score computation.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        id: int
                Node index
        true_vec: np.array
                The true_vec array has the same length as X. It is equal to 1 in correspondance of the nodes
                where a sample passed, 0 otherwise.
                By default the value of true_vec is None
        Returns
        ----------
        Returns true_vec summed with two recursive calls, one on the left son and one on the right son.
        """
        if id == 0:
            true_vec = np.ones(len(X))
        s = self.nodes[id]["point"]
        n = self.nodes[id]["normal"]

        if s is None:
            return true_vec * 1
        else:
            val = np.array(X[true_vec == 1].dot(n) - s > 0)
            lefts = true_vec.copy()
            rights = true_vec.copy()
            lefts[true_vec == 1] = val
            rights[true_vec == 1] = np.logical_not(val)
            return (
                true_vec * 1
                + self.compute_paths2(X, int(self.left_son[id]), true_vec=lefts)
                + self.compute_paths2(X, int(self.right_son[id]), true_vec=rights)
            )

    def compute_paths(self, X: np.ndarray):
        """
        Alternative method to compute the path followed by sample from the root to the leaf where it is contained. The result of this
        function is used for the Anomaly Score computation.
        This function is the iterative version of the compute_paths2 function.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
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
                s = self.nodes[id]["point"]
                if s is None:
                    break
                n = self.nodes[id]["normal"]
                if x.dot(n) - s > 0:
                    id = self.left_son[id]
                else:
                    id = self.right_son[id]
                k += 1
            return k

        return np.apply_along_axis(path, 1, X)

    """ 
    def predict(self,X, algorithm = 1):
        mean_path = np.zeros(len(X))
        if algorithm == 1:
            for i in self.forest:
                mean_path+=i.compute_paths(X)
                
        elif algorithm == 0:
            for i in self.forest:
                mean_path+=i.compute_paths2(X,0)

        mean_path = mean_path/len(self.forest)
        c = c_factor(len(X))
            
        return 2**(-mean_path/c)
    """


class ExtendedIF:
    """
    EIF/EIF_plus model implementation
    --------------------------------------------------------------------------------

    Parameters
    ----------
    n_trees: int
            Number of Isolation Trees composing the forest
    max_depth: int
            Maximum depth at which a sample can be considered as isolated
    min_sample: int
            Minimum number of samples in a node where isolation is achieved
    dims: int
            Number of degrees of freedom used in the separating hyperplanes.
    subsample_size: int
            Subsample size used in each Isolation Tree
    forest: List
            List of objects from the class ExtendedTree
    plus: int
            This parameter is used to distinguish the EIF and the EIF_plus models. If plus=0 then the EIF model
            will be used, if plus=1 than the EIF_plus model will be considered.
    """

    def __init__(
        self,
        n_trees,
        max_depth=None,
        min_sample=None,
        dims=None,
        subsample_size=None,
        plus=1,
        num_processes_anomaly=1,
        disable_fit_tqdm=True,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.dims = dims
        self.subsample_size = subsample_size
        self.forest = None
        self.plus = plus
        self.num_processes_anomaly = num_processes_anomaly
        self.disable_fit_tqdm = disable_fit_tqdm

    def fit(self, X):
        """
        Fit the EIF/EIF_plus model.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        Returns
        ----------

        """
        if not self.dims:
            self.dims = X.shape[1]
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = np.inf

        self.forest = [
            ExtendedTree(self.dims, self.min_sample, self.max_depth, self.plus)
            for i in range(self.n_trees)
        ]

        if self.subsample_size:
            # crea direttamente la "selezione" del subset random choice of shape (num_trees, subsample_size)
            subsets_idxs = np.random.randint(
                0, X.shape[0], size=(self.n_trees, self.subsample_size)
            )

        for i, x in tqdm(
            enumerate(self.forest), disable=self.disable_fit_tqdm, desc="Fitting forest"
        ):
            if not self.subsample_size:
                x.make_tree(X.view(), 0, 0)
            else:
                X_sub = X.view()[subsets_idxs[i], :]
                x.make_tree(X_sub, 0, 0)

        # print("average number of nodes:", np.mean([len(tree.nodes) for tree in self.forest]))
        # print("std number of nodes:", np.std([len(tree.nodes) for tree in self.forest]))

    @staticmethod
    def segment_sum(segment: list[ExtendedTree], X):
        part_sum = np.sum([tree.compute_paths(X) for tree in segment], axis=0)
        return part_sum

    def Anomaly_Score(self, X, algorithm=1):
        """
        Compute the Anomaly Score for an input dataset
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        algorithm: int
                This variable is used to decide weather to use the compute_paths or compute_paths2 function in the computation of the
                Anomaly Scores.
        Returns
        ----------
        Returns the Anomaly Scores of all the samples contained in the input dataset X.
        """
        mean_path = np.zeros(len(X))
        assert (
            self.forest is not None
        ), "The model has not been fitted yet. Please call the fit function before using the Anomaly_Score function"

        if algorithm == 1:
            if self.num_processes_anomaly > 1:
                # divide the self.forest list into segments
                segment_size = len(self.forest) // self.num_processes_anomaly
                segment_size = max(segment_size, 1)

                segments = [
                    self.forest[i : i + segment_size]
                    for i in range(0, len(self.forest), segment_size)
                ]

                partial_sum = partial(self.segment_sum, X=X)

                with Pool(self.num_processes_anomaly) as pool:
                    mean_path = pool.map(partial_sum, segments)
                    mean_path = sum(mean_path)
            else:
                for i in self.forest:
                    mean_path += i.compute_paths(X)

        elif algorithm == 0:
            for i in self.forest:
                mean_path += i.compute_paths2(X, 0)

        mean_path = mean_path / len(self.forest)
        c = c_factor(len(X))

        return 2 ** (-mean_path / c)

    def _predict(self, X, p):
        """
        Predict the anomalous or not anomalous nature of a set of input samples
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        p: int
                Contamination factor used to determine the threshold to apply to the Anomaly Score for the prediction
        Returns
        ----------
        y_hat: np.array
                Returns 0 for inliers and 1 for outliers
        """
        An_score = self.Anomaly_Score(X)
        y_hat = An_score > sorted(An_score, reverse=True)[int(p * len(An_score))]
        return y_hat

    # ???
    def evaluate(self, X, y, p):
        An_score = self.Anomaly_Score(X)
        m = np.c_[An_score, y]
        m = m[(-m[:, 0]).argsort()]
        return np.sum(m[: int(p * len(X)), 1]) / int(p * len(X))

    def print_score_map(self, X, resolution, plot=None, features=[0, 1]):
        """
        Produce the Anomaly Score Scoremap.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        resolution: int
                Scoremap resolution
        plot: None
                Variable used to distinguish different ways of managing the plot settings.
                By default the value of plot is set to None.
        features: List
                List containing the pair of variables compared in the Scoremap.
                By default the value of features is [0,1].
        Returns
        ----------
        Returns the Anomaly Score Scoremap
        """
        if plot == None:
            fig, plot = plt.subplots(1, 1, figsize=(10, 10))
        mins = X[:, features].min(axis=0)
        maxs = X[:, features].max(axis=0)
        mins = list(mins - (maxs - mins) * 3 / 10)
        maxs = list(maxs + (maxs - mins) * 3 / 10)
        xx, yy = np.meshgrid(
            np.linspace(mins[0], maxs[0], resolution),
            np.linspace(mins[1], maxs[1], resolution),
        )

        means = np.mean(X, axis=0)
        feat_0 = xx.ravel()
        feat_1 = yy.ravel()
        dataset = np.array([x * np.ones(len(feat_0)) for x in means]).T
        dataset[:, features[0]] = feat_0
        dataset[:, features[1]] = feat_1
        S1 = self.Anomaly_Score(dataset)
        S1 = S1.reshape(xx.shape)
        x = X.T[0]
        y = X.T[1]

        levels = np.linspace(np.min(S1), np.max(S1), 10)
        CS = plot.contourf(xx, yy, S1, levels, cmap=plt.cm.YlOrRd)
        cb = colorbar(CS, extend="max")
        cb.ax.set_yticks(fontsize=12)
        cb.ax.set_ylabel("Anomaly Score", fontsize=16)
        plot.scatter(x, y, s=15, c="None", edgecolor="k")
        plot.set_title(
            "Anomaly Score scoremap with {} Degree of Freedom".format(self.dims),
            fontsize=18,
        )
        plot.set_xlabel("feature {}".format(features[0]), fontsize=16)
        plot.set_ylabel("feature {}".format(features[1]), fontsize=16)


class ExtendedTree_c:
    def __init__(self, dims: int, min_sample: int, max_depth: int, plus: int):
        """
        Implementation of Isolation Trees for the EIF/EIF_plus models
        --------------------------------------------------------------------------------

        Parameters
        ----------
        dims: int
                Number of degrees of freedom used in the separating hyperplanes
        min_sample: int
                Minimum number of samples in a node where isolation is achieved
        max_depth: int
                Maximum depth at which a sample can be considered as isolated
        plus: int
                This parameter is used to distinguish the EIF and the EIF_plus models. If plus=0 then the EIF model
                will be used, if plus=1 than the EIF_plus model will be considered.
        """
        self.dims = dims
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.depth = 0
        self.right_son = [0]
        self.left_son = [0]
        self.nodes = {}
        self.plus = plus

    @staticmethod
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
        for i, node in nodes.items():
            dest_arr[i].numerosity = ctypes.c_int(node["numerosity"])
            dest_arr[i].is_leaf = ctypes.c_bool(node["point"] is None)
            # allocate the memory for the normal array
            dest_arr[i].normal = (ctypes.c_double * num_features)()
            if node["point"] is None:
                continue
            else:
                dest_arr[i].point = ctypes.c_double(node["point"])
                dest_arr[i].normal = (ctypes.c_double * num_features)(*node["normal"])

    def convert_to_c_data(self):
        # --- Create the data structures for the C function ---
        # convert the nodes dictionary to a ctypes array of Node structure instances
        NodeArray = Node * len(self.nodes)
        c_node_arr = NodeArray()

        if self.dims:
            X_cols = self.dims
        else:
            X_cols = len(self.nodes[0]["normal"])

        ExtendedTree_c.nodes_to_c_array(self.nodes, c_node_arr, X_cols)

        # convert the lists in numpy arrays
        c_left_son = np.array(self.left_son, dtype=np.int32)
        c_right_son = np.array(self.right_son, dtype=np.int32)

        self.c_node_arr = c_node_arr
        self.c_left_son = c_left_son
        self.c_right_son = c_right_son

    def make_tree(self, X: np.ndarray, id: int, depth: int):
        """
        Create an Isolation Tree using the separating hyperplanes
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        id: int
                Node index
        depth: int
                Depth value

        N.B The id and depth input parameters are needed because this is a recursive algorithm. At the first call id and depth
        will be equal to 0.

        """
        if X.shape[0] <= self.min_sample or depth >= self.max_depth:
            # reached leaf
            self.nodes[id] = {"point": None, "normal": None, "numerosity": len(X)}
        else:
            n = make_rand_vector(self.dims, X.shape[1])

            val = X.dot(n)
            s = (
                np.random.normal(np.mean(val), np.std(val) * 2)
                if np.random.random() < self.plus
                else np.random.uniform(np.min(val), np.max(val))
            )
            lefts = val > s

            self.nodes[id] = {"point": s, "normal": n, "numerosity": len(X)}

            idsx = len(self.nodes)
            self.left_son[id] = int(idsx)
            self.right_son.append(0)
            self.left_son.append(0)
            self.make_tree(X[lefts], idsx, depth + 1)

            iddx = len(self.nodes)
            self.right_son.append(0)
            self.left_son.append(0)
            self.right_son[id] = int(iddx)
            self.make_tree(X[~lefts], iddx, depth + 1)

    def compute_paths(
        self,
        X: npt.NDArray[np.float64],
        X_shape: tuple[int, int],
        paths: Optional[npt.NDArray[np.int32]] = None,
    ):
        """
        Alternative method to compute the path followed by sample from the root to the leaf where it is contained.
        The result of this function is used for the Anomaly Score computation.

        Parameters
            X: flattened numpy array of the input dataset
            c_X_cols: number of columns of the input dataset
            c_X_rows: number of rows of the input dataset

        Returns
            paths: np array of nodes encountered by a sample in its path towards the leaf in which it is contained.
        """
        c_X_rows, c_X_cols = ctypes.c_int(X_shape[0]), ctypes.c_int(X_shape[1])

        if not paths:
            paths = np.zeros(X_shape[0], dtype=np.int32)

        c_compute_paths(
            X,
            self.c_node_arr,
            self.c_left_son,
            self.c_right_son,
            c_X_rows,
            c_X_cols,
            paths,
        )

        return paths


class ExtendedIF_c:
    """
    EIF/EIF_plus model implementation
    --------------------------------------------------------------------------------

    Parameters
    ----------
    n_trees: int
            Number of Isolation Trees composing the forest
    max_depth: int
            Maximum depth at which a sample can be considered as isolated
    min_sample: int
            Minimum number of samples in a node where isolation is achieved
    dims: int
            Number of degrees of freedom used in the separating hyperplanes.
    subsample_size: int
            Subsample size used in each Isolation Tree
    forest: List
            List of objects from the class ExtendedTree
    plus: int
            This parameter is used to distinguish the EIF and the EIF_plus models. If plus=0 then the EIF model
            will be used, if plus=1 than the EIF_plus model will be considered.
    """

    def __init__(
        self,
        n_trees,
        max_depth=None,
        min_sample=None,
        dims=None,
        subsample_size=None,
        plus=1,
        num_processes_anomaly=1,
        disable_fit_tqdm=True,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.dims = dims
        self.subsample_size = subsample_size
        self.forest = None
        self.plus = plus
        self.num_processes_anomaly = num_processes_anomaly
        self.disable_fit_tqdm = disable_fit_tqdm

    def fit(self, X):
        """
        Fit the EIF/EIF_plus model.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        Returns
        ----------

        """
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = sys.maxsize
        if not self.dims:
            self.dims = X.shape[1]

        self.forest = [
            ExtendedTree_c(self.dims, self.min_sample, self.max_depth, self.plus)
            for _ in range(self.n_trees)
        ]

        if self.subsample_size:
            # crea direttamente la "selezione" del subset random choice of shape (num_trees, subsample_size)
            subsets_idxs = np.random.randint(
                0, X.shape[0], size=(self.n_trees, self.subsample_size)
            )

        for i, x in tqdm(
            enumerate(self.forest), disable=self.disable_fit_tqdm, desc="Fitting forest"
        ):
            if not self.subsample_size:
                x.make_tree(X.view(), 0, 0)
            else:
                X_sub = X.view()[subsets_idxs[i], :]
                x.make_tree(X_sub, 0, 0)

            x.convert_to_c_data()

    def c_AnomalyScore(self, X: npt.NDArray):
        assert self.forest is not None, "The model has not been fitted yet."
        # dataset to be used in the C function
        c_X = X.astype(np.float64).flatten()
        # convert the nodes array to a ctypes array of Node structure instances
        node_lst = [tree.c_node_arr for tree in self.forest]
        c_node_arr = (ctypes.POINTER(Node) * len(node_lst))(*node_lst)
        # left son array
        left_son_lst = [
            tree.c_left_son.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            for tree in self.forest
        ]
        c_left_son = (ctypes.POINTER(ctypes.c_int) * len(left_son_lst))(*left_son_lst)
        # right son array
        right_son_lst = [
            tree.c_right_son.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            for tree in self.forest
        ]
        c_right_son = (ctypes.POINTER(ctypes.c_int) * len(right_son_lst))(
            *right_son_lst
        )
        # num trees
        num_trees = ctypes.c_int(int(len(self.forest)))
        # num rows
        num_rows = ctypes.c_int(int(X.shape[0]))
        # num cols
        num_cols = ctypes.c_int(int(X.shape[1]))
        # result array
        anomaly_scores = np.zeros(len(X), dtype=np.float64)

        c_anomaly_score(
            c_X,
            c_node_arr,
            c_left_son,
            c_right_son,
            num_trees,
            num_rows,
            num_cols,
            anomaly_scores,
        )

        return anomaly_scores

    def paths2anomaly_score(self, sum_paths, num_samples):
        """
        Given the sum of all paths across the trees in the forest, compute the anomaly score.
        """
        assert self.forest is not None, "The model has not been fitted yet."
        mean_paths = sum_paths / len(self.forest)
        c = c_factor(num_samples)
        return 2.0 ** (-mean_paths / c)

    def Anomaly_Score(self, X):
        """
        Compute the Anomaly Score for an input dataset
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        Returns
        ----------
        Returns the Anomaly Scores of all the samples contained in the input dataset X.
        """
        sum_paths = np.zeros(len(X))
        assert (
            self.forest is not None
        ), "The model has not been fitted yet. Please call the fit function before using the Anomaly_Score function"

        c_X = X.astype(np.float64).flatten()

        for tree in self.forest:
            sum_paths += tree.compute_paths(c_X, X.shape)

        return self.paths2anomaly_score(sum_paths, len(X))

    def _predict(self, X, p):
        """
        Predict the anomalous or not anomalous nature of a set of input samples
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        p: int
                Contamination factor used to determine the threshold to apply to the Anomaly Score for the prediction
        Returns
        ----------
        y_hat: np.array
                Returns 0 for inliers and 1 for outliers
        """
        An_score = self.Anomaly_Score(X)
        y_hat = An_score > sorted(An_score, reverse=True)[int(p * len(An_score))]
        return y_hat

    # ???
    def evaluate(self, X, y, p):
        An_score = self.Anomaly_Score(X)
        m = np.c_[An_score, y]
        m = m[(-m[:, 0]).argsort()]
        return np.sum(m[: int(p * len(X)), 1]) / int(p * len(X))

    def print_score_map(self, X, resolution, plot=None, features=[0, 1]):
        """
        Produce the Anomaly Score Scoremap.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        resolution: int
                Scoremap resolution
        plot: None
                Variable used to distinguish different ways of managing the plot settings.
                By default the value of plot is set to None.
        features: List
                List containing the pair of variables compared in the Scoremap.
                By default the value of features is [0,1].
        Returns
        ----------
        Returns the Anomaly Score Scoremap
        """
        if plot == None:
            fig, plot = plt.subplots(1, 1, figsize=(10, 10))
        mins = X[:, features].min(axis=0)
        maxs = X[:, features].max(axis=0)
        mins = list(mins - (maxs - mins) * 3 / 10)
        maxs = list(maxs + (maxs - mins) * 3 / 10)
        xx, yy = np.meshgrid(
            np.linspace(mins[0], maxs[0], resolution),
            np.linspace(mins[1], maxs[1], resolution),
        )

        means = np.mean(X, axis=0)
        feat_0 = xx.ravel()
        feat_1 = yy.ravel()
        dataset = np.array([x * np.ones(len(feat_0)) for x in means]).T
        dataset[:, features[0]] = feat_0
        dataset[:, features[1]] = feat_1
        S1 = self.Anomaly_Score(dataset)
        S1 = S1.reshape(xx.shape)
        x = X.T[0]
        y = X.T[1]

        levels = np.linspace(np.min(S1), np.max(S1), 10)
        CS = plot.contourf(xx, yy, S1, levels, cmap=plt.cm.YlOrRd)
        cb = colorbar(CS, extend="max")
        cb.ax.set_yticks(fontsize=12)
        cb.ax.set_ylabel("Anomaly Score", fontsize=16)
        plot.scatter(x, y, s=15, c="None", edgecolor="k")
        plot.set_title(
            "Anomaly Score scoremap with {} Degree of Freedom".format(self.dims),
            fontsize=18,
        )
        plot.set_xlabel("feature {}".format(features[0]), fontsize=16)
        plot.set_ylabel("feature {}".format(features[1]), fontsize=16)
