#ifndef COMMON_H
#define COMMON_H

struct Node
{
    double point;   // scalar offset along the normal to identify the hiperplane cut
    double *normal; // normal of the hyperplane that cuts the space in 2 (len = num features)
    int numerosity; // number of samples in this half of the space
    bool is_leaf;   // leaves do not have "normals" because they don't have a plane to cut the space and neither an offset
};

inline double dot_product(double *a, double *b, int n)
{
    double result = 0;
    // #pragma omp parallel for reduction(+ : result) // not worth it
    for (int i = 0; i < n; i++)
        result += a[i] * b[i];

    return result;
}



#endif // COMMON_H