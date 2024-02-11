// To compile:
// $ gcc -fopenmp -O2 -fPIC -shared -o c_compute_paths.so c_compute_paths.c

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
 * @return Void. The computed paths are stored in the given computed_paths array.
 */
void compute_paths(
    double *X,
    struct Node *nodes,
    int *left_son,
    int *right_son,
    int X_rows,
    int X_cols, // number of features
    int *computed_paths)
{
#pragma omp parallel for
    for (int i = 0; i < X_rows; i++)
    {
        int id = 0, k = 1;
        while (true)
        {
            if (nodes[id].is_leaf)
                break;
            // dot product between x sample and current normal vector
            double dot = dot_product(&X[i * X_cols], nodes[id].normal, X_cols);
            // compute which side of the hyperplane the sample is
            id = (dot - nodes[id].point > 0) ? left_son[id] : right_son[id];
            k++;
        }
        computed_paths[i] = k;
    }
}
