#include <stdbool.h>
#include "common.h"


/**
 * @brief
 *
 * @param X: flattened input dataset
 * @param nodes: array of instances of the Node struct
 * @param left_son: array of left sons ids of each node
 * @param right_son: array of right sons ids of each node
 * @param X_rows: number of rows of X, which is the number of SAMPLES in the dataset
 * @param X_cols: number of columns of X, which is the number of FEATURES in the dataset
 *
 * @return Void.
 */
void compute_paths(
    double *X,
    struct Node *nodes,
    bool depth_based,
    int *left_son,
    int *right_son,
    int num_nodes,
    int X_rows,
    int X_cols, // number of features
    double *Importances_list,
    double *Normal_vectors_list)
{
    const int len = X_cols * X_rows;
#pragma omp parallel for shared(X, nodes, left_son, right_son) schedule(dynamic)
    for (int i = 0; i < X_rows; i++)
    {
        double *curr_X = X + i * X_cols;

        int id = 0;
        while (true) // add a max number of iterations to avoid infinite loops?
        {
            // if this is a leaf, stop
            if (nodes[id].is_leaf)
                break;

            double *normal = nodes[id].normal;

            // dot product between x sample and current normal vector
            double dot = dot_product(curr_X, normal, X_cols);

            if (dot - nodes[id].point > 0)
                id = left_son[id];
            else
                id = right_son[id];

            // update the importance and normal vectors arrays
            single_imp_and_abs_vec(
                normal,
                curr_imp,
                curr_norm,
                X_cols,
                father_numerosity,
                nodes[id].numerosity,
                depth_based,
                depth);

            depth++;
        }
    }
}