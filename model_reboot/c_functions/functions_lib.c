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
        // the importance is given by the abs(normal),
        // but we also account for the number of
        // importance = abs(normal) * total_num_samples / num_samples_in_split
        double partial_importance = fabs(node->normal[i]);
        cumul_normal[i] += partial_importance;
        partial_importance *= mask_len; // mask_len is the same as node_size
        int true_count = count_true_in_arr(mask, mask_len);
        l_cumul_importance[i] = cumul_importance[i] + partial_importance / (true_count + 1);
        r_cumul_importance[i] = cumul_importance[i] + partial_importance / (mask_len - true_count + 1);
    }
}

void get_centroid(int vec_len, double *samples, int num_samples, double *centroid)
{
    for (int i = 0; i < vec_len; i++)
    {
        centroid[i] = 0;
        for (int j = 0; j < num_samples; j++)
        {
            centroid[i] += samples[j * vec_len + i];
        }
        centroid[i] /= num_samples;
    }
}

void print_buffer(double *buffer, int len)
{
    for (int i = 0; i < len; i++)
    {
        printf("%f ", buffer[i]);
    }
    printf("\n");
}

// NOTE, TO OPTIMIZE:
//  - left_subset_len should be equal to the number of true values in the mask
//  - right_subset_len should be equal to the number of false values in the mask
//  - left_subset_samples should be accessed from the dataset, so we don't need to pass it
//  - right_subset_samples should be accessed from the dataset, so we don't need to pass it
void update_importances_and_normals_centroid(struct Node *node,
                                             int vec_len,
                                             double *cumul_normal,
                                             double *cumul_importance,
                                             double *l_cumul_importance,
                                             double *r_cumul_importance,
                                             bool *mask,
                                             int mask_len,
                                             double *dataset, // flattened
                                             int dataset_len,
                                             double *left_subset_samples,
                                             int left_subset_len,
                                             double *right_subset_samples,
                                             int right_subset_len)
{
    // the importance is given by the abs(distance between the centroids),
    // where the distance is the vector connecting the two centroids,
    // and the centroids are the mean of the left and right subset samples,
    // but if either the left or right subset is empty, use the normal as the distance
    double *centroid_distance = (double *)malloc(vec_len * sizeof(double));
    
    if (left_subset_len == 0 || right_subset_len == 0)
    { 
        for (int i = 0; i < vec_len; i++)
        {
            centroid_distance[i] = node->normal[i];
        }
    }
    else
    {
        // compute the first centroid as the sum of the left subset samples
        double *centroid1 = (double *)malloc(vec_len * sizeof(double));
        get_centroid(vec_len, left_subset_samples, left_subset_len, centroid1);
        // compute the second centroid as the sum of the right subset samples
        double *centroid2 = (double *)malloc(vec_len * sizeof(double));
        get_centroid(vec_len, right_subset_samples, right_subset_len, centroid2);
        // compute the distance between the centroids
        for (int i = 0; i < vec_len; i++)
        {
            centroid_distance[i] = centroid1[i] - centroid2[i];
        }
        free(centroid1);
        free(centroid2);
    }

    for (int i = 0; i < vec_len; i++)
    {
        // distance between the centroids
        double partial_importance = fabs(centroid_distance[i]);

        // but we also account for the number of
        // importance = abs(importance direction) * total_num_samples / num_samples_in_split

        cumul_normal[i] += partial_importance;
        partial_importance *= mask_len; // mask_len is the same as node_size
        int true_count = count_true_in_arr(mask, mask_len);
        l_cumul_importance[i] = cumul_importance[i] + partial_importance / (true_count + 1);
        r_cumul_importance[i] = cumul_importance[i] + partial_importance / (mask_len - true_count + 1);
    }

    free(centroid_distance);
}
