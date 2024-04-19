#include <stdlib.h>
#include <stdbool.h>

struct LeafData
{
    // Define the fields of LeafData here
};

struct Node
{
    double intercept;
    double *normal;
    unsigned int left_child_id;
    unsigned int right_child_id;
    struct LeafData *leaf_data;
    unsigned int id;
    bool is_leaf;
};

/**
 * @brief Dot product between vector a of shape (shape0, shape1) and vector b of shape (shape1,)
*/
void dot_broadcast(double *a, double *b, int shape0, int shape1, double *res)
{
    // #pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < shape0; i++)
    {
        res[i] = 0;
        for (int j = 0; j < shape1; j++)
        {
            res[i] += a[i * shape1 + j] * b[j];
        }
    }
}

double *copy_alloc(double *a, int len)
{
    double *res = (double *)malloc(len * sizeof(double));
    for (int i = 0; i < len; i++)
    {
        res[i] = a[i];
    }
    return res;
}

inline double dot(double *a, double *b, int len)
{
    double result = 0;
    for (int i = 0; i < len; i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Get the leaf ids of the dataset samples
 *
 * @param dataset The dataset samples, flattened
 * @param num_rows The number of samples in the dataset
 * @param num_cols The number of features in the dataset
 * @param dest_leaf_ids The destination array to store the leaf ids
 * @param nodes The nodes of the tree
 * @param n_nodes The number of nodes in the tree
 */
void get_leaf_ids(double *dataset,
                  int num_rows,
                  int num_cols,
                  int *dest_leaf_ids,
                  struct Node **nodes,
                  int n_nodes)
{
#pragma omp parallel for shared(dataset) shared(dest_leaf_ids) shared(nodes) schedule(dynamic, 1)
    for (int i = 0; i < num_rows; i++)
    {
        struct Node node = *nodes[0];
        while (!node.is_leaf)
        {
            double *x = &dataset[i * num_cols];
            double d = dot(x, node.normal, num_cols);
            int node_id = (d <= node.intercept) ? node.left_child_id : node.right_child_id;
            node = *nodes[node_id];
        }
        dest_leaf_ids[i] = node.id;
    }
}
