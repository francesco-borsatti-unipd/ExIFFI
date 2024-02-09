import sys
import pickle
from typing import List
import logging
from functools import partial
from multiprocessing import Pool, cpu_count


sys.path.append("./models")
from models.Extended_IF import ExtendedIF, ExtendedTree
import numpy as np


class Extended_DIFFI_tree(ExtendedTree):
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

    def make_importance(self, X, depth_based):
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

        # multicore processing
        num_processes = 1  ####################################################

        if num_processes > 1:
            partial_importance_worker = partial(
                self.importance_worker,
                nodes=self.nodes,
                depth_based=depth_based,
                left_son=self.left_son,
                right_son=self.right_son,
            )

            # split the input vector num_processes
            segment_size = len(X) // num_processes
            segments = [X[i : i + segment_size] for i in range(0, len(X), segment_size)]

            with Pool(processes=num_processes) as pool:
                results = pool.map(partial_importance_worker, segments)

                Importances_list = []
                Normal_vectors_list = []

                for result in results:
                    Importances_list.extend(result[0])
                    Normal_vectors_list.extend(result[1])
        else:
            Importances_list, Normal_vectors_list = self.importance_worker(
                X, self.nodes, depth_based, self.left_son, self.right_son
            )

        return np.array(Importances_list), np.array(Normal_vectors_list)


class Extended_DIFFI_parallel(ExtendedIF):
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)
        self.sum_importances_matrix = None
        self.sum_normal_vectors_matrix = None
        self.plus = kwarg.get("plus")
        self.seed = kwarg.get("seed")
        self.num_processes_importances = 1
        self.num_processes_fit = 1
        self.num_processes_anomaly = 1

    @staticmethod
    def make_tree_worker(forest_segment: List[Extended_DIFFI_tree], X, subsample_size):
        # subsets = []
        for x in forest_segment:
            if not subsample_size or subsample_size > X.shape[0]:
                x.make_tree(X, 0, 0)
            else:
                indx = np.random.choice(X.shape[0], subsample_size, replace=False)
                X_sub = X[indx, :]
                x.make_tree(X_sub, 0, 0)
                # subsets.append(indx)
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

        if hasattr(self, "num_fit_calls"):
            self.num_fit_calls += 1
        else:
            # first call
            self.num_fit_calls = 0

        self.forest = [
            Extended_DIFFI_tree(
                dims=self.dims,
                min_sample=self.min_sample,
                max_depth=self.max_depth,
                plus=self.plus,
                seed=(
                    self.seed + i + self.num_fit_calls * self.n_trees * self.max_depth
                    if self.seed
                    else None
                ),
            )
            for i in range(self.n_trees)
        ]

        if self.num_processes_fit > 1:
            # --- Parallel execution ---
            partial_make_tree_worker = partial(
                self.make_tree_worker, X=X, subsample_size=self.subsample_size
            )

            segment_size = max(1, len(self.forest) // self.num_processes_fit)

            segments = [
                self.forest[i : i + segment_size]
                for i in range(0, len(self.forest), segment_size)
            ]
            with Pool(processes=self.num_processes_fit) as pool:
                results = pool.map(partial_make_tree_worker, segments)

                self.forest = []
                for result in results:
                    self.forest.extend(result)

        else:
            # --- Serial execution ---
            self.subsets = []
            for x in self.forest:
                if not self.subsample_size or self.subsample_size > X.shape[0]:
                    x.make_tree(X, 0, 0)
                else:
                    indx = np.random.choice(
                        X.shape[0], self.subsample_size, replace=False
                    )
                    X_sub = X[indx, :]
                    x.make_tree(X_sub, 0, 0)
                    self.subsets.append(indx)

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

    @staticmethod
    def forest_worker(forest: List[Extended_DIFFI_tree], X, depth_based):
        """
        This takes a segment of the forest, which is a list of trees.
        Given a dataset X and the depth_based option, return a function that
        computes the sum of the importance scores and the sum of the normal vectors.
        """
        # forest, X, depth_based = args

        partial_sum_importances_matrix = np.zeros_like(X, dtype="float64")
        partial_sum_normal_vectors_matrix = np.zeros_like(X, dtype="float64")

        for tree in forest:
            importances_matrix, normal_vectors_matrix = tree.make_importance(
                X, depth_based
            )
            partial_sum_importances_matrix += importances_matrix
            partial_sum_normal_vectors_matrix += normal_vectors_matrix

        return partial_sum_importances_matrix, partial_sum_normal_vectors_matrix

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
        if (self.sum_importances_matrix is None) or calculate:
            sum_importances_matrix = np.zeros_like(X, dtype="float64")
            sum_normal_vectors_matrix = np.zeros_like(X, dtype="float64")

            # multicore processing
            # split the input vector into segments
            # from this: [tree0, tree1, tree2, tree3, tree4]
            # to this: [  [tree0, tree1],   [tree2, tree3], [tree4]]]

            segment_size = max(1, len(self.forest) // self.num_processes_importances)

            print("self.num_processes_importances:", self.num_processes_importances)
            print("segment_size:", segment_size)

            segments = [
                self.forest[i : i + segment_size]
                for i in range(0, len(self.forest), segment_size)
            ]

            print(f"Segments shapes: {[np.array(s).shape for s in segments]}")

            if self.num_processes_importances > 1:
                with Pool(processes=self.num_processes_importances) as pool:
                    forest_worker_partial = partial(
                        self.forest_worker, X=X, depth_based=depth_based
                    )

                    # the result list of tuples which are the outputs of the make_importance function
                    results = pool.map(forest_worker_partial, segments)

                    # results = [ (part_sum_mat0, part_sum_norm0), # worker 0
                    #             (part_sum_mat1, part_sum_norm1), # worker 1
                    #             ...
                    #           ]

                    for result in results:
                        sum_importances_matrix += result[0]
                        sum_normal_vectors_matrix += result[1]

                    # the division can be done at the end
                    sum_importances_matrix /= self.n_trees
                    sum_normal_vectors_matrix /= self.n_trees
            else:
                print("Forest worker is serial")
                sum_importances_matrix, sum_normal_vectors_matrix = self.forest_worker(
                    self.forest, X, depth_based
                )

            if overwrite:
                self.sum_importances_matrix = sum_importances_matrix / self.n_trees
                self.sum_normal_vectors_matrix = (
                    sum_normal_vectors_matrix / self.n_trees
                )

            return sum_importances_matrix, sum_normal_vectors_matrix
        else:
            return self.sum_importances_matrix, self.sum_normal_vectors_matrix

    def Global_importance(self, X, calculate, overwrite, depth_based=False):
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
        print("Start computing Anomaly Score")
        anomaly_scores = self.Anomaly_Score(X)
        print("End computing Anomaly Score")
        ind = np.argpartition(anomaly_scores, -int(0.1 * len(X)))[-int(0.1 * len(X)) :]

        print("Start computing Importances Score")
        importances_matrix, normal_vectors_matrix = self.Importances(
            X, calculate, overwrite, depth_based
        )
        print("Stop computing Importances Score")

        Outliers_mean_importance_vector = np.mean(importances_matrix[ind], axis=0)
        Inliers_mean_Importance_vector = np.mean(
            importances_matrix[np.delete(range(len(importances_matrix)), ind)], axis=0
        )

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
        importances_matrix, normal_vectors_matrix = self.Importances(
            X, calculate, overwrite, depth_based
        )
        return importances_matrix / normal_vectors_matrix

