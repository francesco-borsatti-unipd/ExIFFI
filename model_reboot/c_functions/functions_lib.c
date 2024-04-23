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
                cumul_importances[i * vec_len + j] = nodes[i]->leaf_data->cumul_importance[j];
                cumul_normals[i * vec_len + j] = nodes[i]->leaf_data->cumul_normals[j];
            }
        }
    }
}

/**
 */
void alloc_child_nodes(struct Node **nodes,
                       int max_nodes,
                       int num_nodes,
                       struct Node *current_node)
{
    if (num_nodes + 2 > max_nodes)
    {
        fprintf(stderr, "Not enough memory to allocate the child nodes\n");
        exit(EXIT_FAILURE);
    }
    // allocate the right and left child nodes for the current node
    struct Node *left_node = (struct Node *)malloc(sizeof(struct Node));
    current_node->left_child_id = left_node->id = num_nodes;
    nodes[num_nodes] = left_node;
    num_nodes += 1;

    struct Node *right_node = (struct Node *)malloc(sizeof(struct Node));
    current_node->right_child_id = right_node->id = num_nodes;
    nodes[num_nodes] = right_node;
}

void save_normal_vector(struct Node *node, double *normal, int vec_len)
{
    // allocate memory for the normal vector
    node->normal = (double *)malloc(vec_len * sizeof(double));
    for (int i = 0; i < vec_len; i++)
    {
        node->normal[i] = normal[i];
    }
}

inline int count_true_in_arr(bool arr[], int size)
{
    int count = 0;
    for (int i = 0; i < size; i++)
    {
        if (arr[i])
        {
            count++;
        }
    }
    return count;
}

/*
Python implementation:

    mask = dist <= node.intercept
    partial_importance = np.abs(
        np.ctypeslib.as_array(node.normal, shape=(X.shape[1],)).astype(
            np.float64
        )
    )
    cumul_normals += partial_importance
    partial_importance *= node_size
    l_cumul_importance = cumul_importance + partial_importance / (
        len(subset_ids[mask]) + 1
    )
    r_cumul_importance = cumul_importance + partial_importance / (
        len(subset_ids[~mask]) + 1
    )
*/

/**
 * @brief Update the importances and normals of the nodes
 *
 * @param mask_len The length of the boolean mask array, its length is equal to the number
 *      of samples in the curent node
 */
void update_importances_and_normals(struct Node *node,
                                    int vec_len,
                                    double *cumul_normal,
                                    double *cumul_importance,
                                    double *l_cumul_importance,
                                    double *r_cumul_importance,
                                    bool *mask,
                                    int mask_len)
{
    for (int i = 0; i < vec_len; i++)
    {
        double partial_importance = fabs(node->normal[i]);
        cumul_normal[i] += partial_importance;
        partial_importance *= mask_len; // mask_len is the same as node_size
        int true_count = count_true_in_arr(mask, mask_len);
        l_cumul_importance[i] = cumul_importance[i] + partial_importance / (true_count + 1);
        r_cumul_importance[i] = cumul_importance[i] + partial_importance / (mask_len - true_count + 1);
    }
}
