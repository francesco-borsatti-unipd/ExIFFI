#include <stdlib.h>


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


double * copy_alloc(double *a, int len)
{
    double *res = (double *)malloc(len * sizeof(double));
    for (int i = 0; i < len; i++)
    {
        res[i] = a[i];
    }
    return res;
}