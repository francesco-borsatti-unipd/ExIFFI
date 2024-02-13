// NOT USED: since the compute_paths is parallel, it does not gain any performance 
//      from being called from a parallel function. 

// To compile:
// $ gcc -fopenmp -O2 -fPIC -shared -o c_anomaly_score.so c_anomaly_score.c

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"

/**
 * @brief Average path length of unsuccesful search in a binary search tree given n points
 * @param n: number data points for the BST
 * @return Average path length of unsuccesful search in a BST
 */
inline double c_factor(int n)
{
    double d_n = (double)n;
    return 2.0 * (log(d_n - 1) + 0.5772156649) - (2.0 * (d_n - 1) / d_n);
}

/**
 * @brief Compute the Anomaly Score for an input dataset, given a forest of trees
 *
 * @param X: flattened input dataset
 * @param forest_nodes: array of arrays of instances of the Node struct
 * @param forest_left_son: array of arrays of left sons ids of each node
 * @param forest_right_son: array of arrays of right sons ids of each node
 * @param num_trees: number of trees in the forest
 * @param X_rows: number of samples in X dataset
 * @param X_cols: number of features in X dataset
 * @param anomaly_scores: destination array of length = num of samples in X dataset (X_rows)
 *
 * @return Void. The computed paths are stored in the given `anomaly_scores` array
 */
void anomaly_score(
    double *X,
    struct Node **forest_nodes,
    int **forest_left_son,
    int **forest_right_son,
    int num_trees,
    int X_rows,
    int X_cols,
    double *anomaly_scores)
{
    // allocate an array of arrays, each of length X_rows, to store the computed paths for each tree
    int *computed_paths = (int *)malloc(X_rows * sizeof(int));
    if (computed_paths == NULL)
    {
        fprintf(stderr, "Error: malloc failed in anomaly_score\n");
        exit(EXIT_FAILURE);
    }
    // initialize the computed_paths array to 0
    for (int i = 0; i < X_rows; i++)
        computed_paths[i] = 0;

#pragma omp parallel for reduction(+ : computed_paths[ : X_rows])
    for (int tree_idx = 0; tree_idx < num_trees; tree_idx++)
    {
        int *this_computed_paths = (int *)malloc(X_rows * sizeof(int));
        if (this_computed_paths == NULL)
        {
            fprintf(stderr, "Error: malloc failed in anomaly_score\n");
            exit(EXIT_FAILURE);
        }

        compute_paths(
            X,
            forest_nodes[tree_idx],
            forest_left_son[tree_idx],
            forest_right_son[tree_idx],
            X_rows,
            X_cols,
            this_computed_paths);

        // add this tree paths for all the points in X to the computed_paths array
        for (int i = 0; i < X_rows; i++)
            computed_paths[i] += this_computed_paths[i];
    }

    double c = c_factor(X_rows);
    for (int i = 0; i < X_rows; i++)
        anomaly_scores[i] = exp2(-(double)computed_paths[i] / (num_trees * c));

    return;
}