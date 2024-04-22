#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

struct LeafData
{
    // Define the fields of LeafData here
    double *cumul_normals;
    double *cumul_importance;
    double corrected_depth;
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

/**
 * @brief Correction factor for the anomaly score computation
 *
 * @param n The number of samples in the current node
 */
double c_factor(int n)
{
    if (n <= 1)
    {
        return 0.0;
    }
    double d_n = (double)n;
    return 2.0 * (log(d_n - 1) + 0.5772156649) - (2.0 * (d_n - 1) / d_n);
}

/**
 * @brief Allocate memory for the leaf data and write it to the node
 *
 * @param node Pointer to the node to write the leaf data to
 */
void save_leaf_data(struct Node *node,
                    double corrected_depth,
                    double *cumul_normals,
                    double *cumul_importance,
                    int vec_len)
{
    node->is_leaf = true;
    node->leaf_data = (struct LeafData *)malloc(sizeof(struct LeafData));
    node->leaf_data->corrected_depth = corrected_depth;

    // Allocate memory for the cumul_normals
    node->leaf_data->cumul_normals = (double *)malloc(vec_len * sizeof(double));
    // Allocate memory for the cumul_importance
    node->leaf_data->cumul_importance = (double *)malloc(vec_len * sizeof(double));
    for (int i = 0; i < vec_len; i++)
    {
        node->leaf_data->cumul_normals[i] = cumul_normals[i];
        node->leaf_data->cumul_importance[i] = cumul_importance[i];
    }
}

/**
 * @brief Retrieve the corrected depths of the leaf nodes and store
 * them in the corrected_depths array. Access only the given leaf ids.
 */
void get_corrected_depths(struct Node **nodes,
                          int num_nodes,
                          double *corrected_depths,
                          int *leaf_ids,
                          int num_leaf_ids)
{
    for (int i = 0; i < num_leaf_ids; i++)
    {
        struct Node *node = nodes[leaf_ids[i]];
        corrected_depths[i] = node->leaf_data->corrected_depth;
    }
}

// memory expensive techjnique
void save_train_data(struct Node **nodes,
                     int num_nodes,
                     double *corrected_depths,
                     double *cumul_importances,
                     double *cumul_normals,
                     int vec_len)
{
    for (int i = 0; i < num_nodes; i++)
    {
        if (nodes[i]->is_leaf)
        {
            corrected_depths[i] = nodes[i]->leaf_data->corrected_depth;
            for (int j = 0; j < vec_len; j++)
            {
                cumul_importances[i*vec_len + j] = nodes[i]->leaf_data->cumul_importance[j];
                cumul_normals[i*vec_len + j] = nodes[i]->leaf_data->cumul_normals[j];
            }
        }
    }
}
