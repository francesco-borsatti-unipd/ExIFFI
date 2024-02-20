import sys, ctypes
from typing import List
from functools import partial
from multiprocessing import Pool
import numpy as np
import numpy.typing as npt
import os


sys.path.append("./models")
from models.Extended_IF import ExtendedIF_c, ExtendedTree_c
from models.c_functions import c_make_importance
from utils.utils import c_factor


class Extended_DIFFI_tree_c(ExtendedTree_c):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.importances = []
        self.sum_normals = []

    @staticmethod
    def importance_worker(X, nodes, depth_based, left_son, right_son):
        """
        Compute the Importance Scores for each node along the Isolation Trees.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.

        Returns
        ----------
        Importances_list: np.array
                List with the Importances values for all the nodes in the Isolation Tree.
        Normal_vectors_list: np.array
                List of all the normal vectors used in the splitting hyperplane creation.
        """
        Importances_list = np.zeros((X.shape[0], X.shape[1]))
        Normal_vectors_list = np.zeros((X.shape[0], X.shape[1]))
        for i, x in enumerate(X):
            id = depth = 0
            while True:
                s = nodes[id]["point"]
                if s is None:
                    break
                n = nodes[id]["normal"]
                N = nodes[id]["numerosity"]
                old_id = id
                if x.dot(n) - s > 0:
                    side = "left_importance"
                    id = left_son[id]
                else:
                    side = "right_importance"
                    id = right_son[id]
                abs_n = np.abs(n)
                singular_importance = abs_n * (N / (nodes[id]["numerosity"] + 1))
                if depth_based == True:
                    singular_importance /= 1 + depth
                Importances_list[i] += singular_importance
                Normal_vectors_list[i] += abs_n
                nodes[old_id].setdefault(side, singular_importance)
                nodes[old_id].setdefault("depth", depth)
                depth += 1
        return Importances_list, Normal_vectors_list

    def make_importance(self, X: npt.NDArray[np.float64], depth_based, X_shape):
        """
        Compute the Importance Scores for each node along the Isolation Trees.

        Parameters
            X (bool): flattened np.array Input dataset of type `float64`
            depth_based (bool): Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance computation function.

        Returns
        Importances_list (np.array): Importances values for all the nodes in the Isolation Tree.
        Normal_vectors_list (np.array): all the normal vectors used in the splitting hyperplane creation.
        """
        X_rows, X_cols = int(X_shape[0]), int(X_shape[1])

        # output vectors
        l = int(X_rows * X_cols)
        paths = np.zeros(X_rows, dtype=np.int32)
        importances = np.zeros(l, dtype=np.float64)
        normal_vectors = np.zeros(l, dtype=np.float64)

        c_make_importance(
            X=X,
            nodes=self.c_node_arr,
            depth_based=ctypes.c_bool(depth_based),
            left_son=self.c_left_son,
            right_son=self.c_right_son,
            X_rows=ctypes.c_int(X_rows),
            X_cols=ctypes.c_int(X_cols),
            paths=paths,
            importances=importances,
            normal_vectors=normal_vectors,
        )
        return paths, importances, normal_vectors


class Extended_DIFFI_c(ExtendedIF_c):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.sum_importances_matrix = None
        self.sum_normal_vectors_matrix = None
        self.plus = kwarg.get("plus")
        self.num_processes_importances = 1
        self.num_processes_fit = 1
        self.num_processes_anomaly = 1

    @staticmethod
    def make_tree_worker(
        forest_segment: List[Extended_DIFFI_tree_c], X, subsample_size
    ):
        # subsets = []
        for x in forest_segment:
            if not subsample_size or subsample_size > X.shape[0]:
                x.make_tree(X, 0, 0)
            else:
                indx = np.random.choice(X.shape[0], subsample_size, replace=False)
                X_sub = X[indx, :]
                x.make_tree(X_sub, 0, 0)
                # subsets.append(indx)
            x.convert_to_c_data()
        return forest_segment

    def fit(self, X):
        """
        Fit the ExIFFI model.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset

        """
        if not self.dims:
            self.dims = X.shape[1]
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = np.inf

        self.forest = [
            Extended_DIFFI_tree_c(
                dims=self.dims,
                min_sample=self.min_sample,
                max_depth=self.max_depth,
                plus=self.plus,
            )
            for i in range(self.n_trees)
        ]

        # --- Serial execution ---
        print("Fitting is serial")
        self.subsets = []
        for x in self.forest:
            if not self.subsample_size or self.subsample_size > X.shape[0]:
                x.make_tree(X, 0, 0)
            else:
                indx = np.random.choice(X.shape[0], self.subsample_size, replace=False)
                X_sub = X[indx, :]
                x.make_tree(X_sub, 0, 0)
                self.subsets.append(indx)
            x.convert_to_c_data()

    def set_num_processes(
        self, num_processes_fit, num_processes_importances, num_processes_anomaly
    ):
        """
        Set the number of processes to be used in the parallel computation
        of the Global and Local Feature Importance.
        """
        self.num_processes_fit = num_processes_fit
        self.num_processes_importances = num_processes_importances
        self.num_processes_anomaly = num_processes_anomaly

    def Importances(self, X, calculate, overwrite, depth_based):
        """
        Obtain the sum of the Importance scores computed along all the Isolation Trees, with the make_importance
        function.
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.
        calculate: bool
                If calculate is True the Importances Sum Matrix and the Normal Vectors Sum Matrix are initialized to 0
        overwrite: bool
                Boolean variable used to decide weather to overwrite evrytime the value inserted in sum_importances_matrix and
                in sum_normal_vectors_matrix.

        Returns
        ----------
        sum_importances_matrix: np.array
                2-dimensional array containing,for each sample, the sum of the importance scores obtained by the nodes in which
                it was included.
        sum_normal_vectors_matrix: np.array
                2-dimensional array containing,for each sample, the sum of the normal vectors used to create the
                splitting hyperplances of the nodes in which it was included.

        """
        if not ((self.sum_importances_matrix is None) or calculate):
            return (
                self.anomaly_scores,
                self.sum_importances_matrix,
                self.sum_normal_vectors_matrix,
            )

        X_shape = X.shape
        X = X.flatten()
        X = X.astype(np.float64)

        sum_paths = np.zeros(X_shape[0], dtype=np.float64)
        sum_importances = np.zeros_like(X, dtype=np.float64)
        sum_normal_vectors = np.zeros_like(X, dtype=np.float64)

        for tree in self.forest:
            paths, importances_matrix, normal_vectors_matrix = tree.make_importance(
                X, depth_based, X_shape
            )
            sum_paths += paths + 1
            sum_importances += importances_matrix
            sum_normal_vectors += normal_vectors_matrix

        # reshape
        sum_importances = sum_importances.reshape(X_shape)
        sum_normal_vectors = sum_normal_vectors.reshape(X_shape)

        # compute anomaly score
        anomaly_scores = self.paths2anomaly_score(sum_paths, X_shape[0])

        if overwrite:
            self.anomaly_scores = anomaly_scores
            self.sum_importances = sum_importances / self.n_trees
            self.sum_normal_vectors = sum_normal_vectors / self.n_trees

        return anomaly_scores, sum_importances, sum_normal_vectors

    def Global_importance(
        self,
        X,
        calculate,
        overwrite,
        depth_based=False,
        imps_and_anomaly_all_in_one=True,
    ):
        """
        Compute the Global Feature Importance vector for a set of input samples
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        calculate: bool
                Used to call the Importances function
        overwrite: bool
                Used to call the Importances function.
         depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.
                By default the value of depth_based is False.

        Returns
        ----------
        Global_Importance: np.array
        Array containig a Global Feature Importance Score for each feature in the dataset.

        """

        print("Start computing Importances Score")
        anomaly_scores, importances_matrix, normal_vectors_matrix = self.Importances(
            X, calculate, overwrite, depth_based
        )
        print("Stop computing Importances Score")

        if imps_and_anomaly_all_in_one:
            print("Anomaly score already computed during importances computation")
        else:
            print("Start computing Anomaly Score")
            anomaly_scores = self.Anomaly_Score(X)
            print("End computing Anomaly Score")

        ind = np.argpartition(anomaly_scores, -int(0.1 * len(X)))[-int(0.1 * len(X)) :]

        Outliers_mean_importance_vector = np.mean(importances_matrix[ind], axis=0)
        Inliers_mean_Importance_vector = np.mean(
            importances_matrix[np.delete(range(len(importances_matrix)), ind)], axis=0
        )

        assert normal_vectors_matrix is not None

        Outliers_mean_normal_vector = np.mean(normal_vectors_matrix[ind], axis=0)
        Inliers_mean_normal_vector = np.mean(
            normal_vectors_matrix[np.delete(range(len(importances_matrix)), ind)],
            axis=0,
        )

        return (Outliers_mean_importance_vector / Outliers_mean_normal_vector) / (
            Inliers_mean_Importance_vector / Inliers_mean_normal_vector
        ) - 1

    def Local_importances(self, X, calculate, overwrite, depth_based=False):
        """
        Compute the Local Feature Importance vector for a set of input samples
        --------------------------------------------------------------------------------

        Parameters
        ----------
        X: pd.DataFrame
                Input dataset
        calculate: bool
                Used to call the Importances function
        overwrite: bool
                Used to call the Importances function.
         depth_based: bool
                Boolean variable used to decide weather to used the Depth Based or the Not Depth Based Importance
                computation function.
                By default the value of depth_based is False.

        Returns
        ----------
        Local_Importance: np.array
        Array containig a Local Feature Importance Score for each feature in the dataset.

        """
        _, importances_matrix, normal_vectors_matrix = self.Importances(
            X, calculate, overwrite, depth_based
        )
        assert normal_vectors_matrix is not None
        return importances_matrix / normal_vectors_matrix
