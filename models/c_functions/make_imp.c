// Normal compilation: gcc -fopenmp -fPIC -shared -o c_make_importance.so make_imp.c
// Optimized compilation: gcc -O2 -fopenmp -fPIC -shared -o c_make_importance.so make_imp.c

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "common.h"

inline void single_imp_and_abs_vec(
    double *normal_vec,
    double *single_importance_arr,
    double *single_normal_vec_arr,
    int len,
    int father_numerosity,
    int son_numerosity,
    bool depth_based,
    int depth)
{
    double depth_reciprocal;
    if (depth_based)
        depth_reciprocal = 1.0 / (double)(depth + 1);

    // #pragma omp parallel for schedule(static) reduction(+:single_importance_arr[:len], single_normal_vec_arr[:len]) // not worth it
    for (int i = 0; i < len; i++)
    {
        double abs_val = fabs(normal_vec[i]);
        double imp = abs_val * ((double)father_numerosity / ((double)son_numerosity + 1.0));
        if (depth_based)
            imp *= depth_reciprocal;

        // atomic operations are not necessary, because the sum is done on different sub-array in each thread
        single_importance_arr[i] += imp;
        single_normal_vec_arr[i] += abs_val;
    }
};

/**
 * @brief Compute the Importance Scores for each node along the Isolation Trees.
 *
 * @param X: flattened input dataset
 * @param nodes: array of instances of the Node struct
 * @param depth_based: boolean to indicate if the importance is depth based
 * @param left_son: array of left sons ids of each node
 * @param right_son: array of right sons ids of each node
 * @param X_rows: number of rows of X, which is the number of SAMPLES in the dataset
 * @param X_cols: number of columns of X, which is the number of FEATURES in the dataset
 *
 * @return Void. The importance scores are stored in the Importances_list and Normal_vectors_list arrays
 */
void importance_worker(
    double *X,
    struct Node *nodes,
    bool depth_based,
    int *left_son,
    int *right_son,
    int X_rows,
    int X_cols, // number of features
    double *Importances_list,
    double *Normal_vectors_list)
{
#pragma omp parallel for shared(X, nodes, left_son, right_son) schedule(dynamic)
    for (int i = 0; i < X_rows; i++)
    {
        double *curr_imp = Importances_list + i * X_cols,
               *curr_norm = Normal_vectors_list + i * X_cols,
               *curr_X = X + i * X_cols;

        int id = 0, father_id = 0, depth = 0;
        while (true) // add a max number of iterations to avoid infinite loops?
        {
            if (nodes[id].is_leaf)
                break;

            // dot product between x sample and current normal vector
            double dot = dot_product(curr_X, nodes[id].normal, X_cols);
            father_id = id;

            // compute which side of the hyperplane the sample is
            id = (dot - nodes[id].point > 0) ? left_son[id] : right_son[id];

            // update the importance and normal vectors arrays
            single_imp_and_abs_vec(
                nodes[father_id].normal,
                curr_imp,
                curr_norm,
                X_cols,
                nodes[father_id].numerosity,
                nodes[id].numerosity,
                depth_based,
                depth);

            depth++;
        }
    }
}