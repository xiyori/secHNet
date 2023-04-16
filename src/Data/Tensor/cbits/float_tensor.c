#include <stdlib.h>
#include <math.h>

#include "cbits.h"

#include <stdio.h>


void
eye_f (
    int rows,
    int columns,
    int k,
    float *dat)
{
    int min_dim = (rows < columns) ? rows : columns;
    int columnShift = (k < 0) ? -k : 0;
    int rowShift = (k > 0) ? k : 0;
    for (int i = 0; i < min_dim - columnShift - rowShift; ++i) {
        dat[(i + columnShift) * columns + i + rowShift] = 1;
    }
}

float
sum_f (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    char *dat)
{
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index = offset;
    float sum = 0, c = 0;
    for (int i = 0; i < numel; ++i) {
        float y = *(float *) (dat + f_index) - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        for (int dim = n_dims - 1; dim >= 0; --dim) {
            index[dim] += 1;
            f_index += stride[dim];
            if (index[dim] == shape[dim]) {
                index[dim] = 0;
                f_index -= stride[dim] * shape[dim];
            } else {
                break;
            }
        }
    }
    free(index);
    return sum;
}

void
map_f (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    char *dat_from,
    char *dat_to,
    float (*f)(float))
{
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index = offset;
    for (int i = 0; i < numel; ++i) {
        *(float *) (dat_to + i * sizeof(float)) = f(
            *(float *) (dat_from + f_index)
        );
        for (int dim = n_dims - 1; dim >= 0; --dim) {
            index[dim] += 1;
            f_index += stride[dim];
            if (index[dim] == shape[dim]) {
                index[dim] = 0;
                f_index -= stride[dim] * shape[dim];
            } else {
                break;
            }
        }
    }
    free(index);
}

float
neg_f(
    float arg)
{
    return -arg;
}

float
sign_f(
    float arg)
{
    return (0 < arg) - (arg < 0);
}

void
elementwise_f (
    int n_dims,
    int *shape,
    int *stride1,
    int offset1,
    char *dat_from1,
    int *stride2,
    int offset2,
    char *dat_from2,
    char *dat_to,
    float (*f)(float, float))
{
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index1 = offset1;
    int f_index2 = offset2;
    for (int i = 0; i < numel; ++i) {
        *(float *) (dat_to + i * sizeof(float)) = f(
            *(float *) (dat_from1 + f_index1),
            *(float *) (dat_from2 + f_index2)
        );
        for (int dim = n_dims - 1; dim >= 0; --dim) {
            index[dim] += 1;
            f_index1 += stride1[dim];
            f_index2 += stride2[dim];
            if (index[dim] == shape[dim]) {
                index[dim] = 0;
                f_index1 -= stride1[dim] * shape[dim];
                f_index2 -= stride2[dim] * shape[dim];
            } else {
                break;
            }
        }
    }
    free(index);
}

float
add_f(
    float arg1,
    float arg2)
{
    return arg1 + arg2;
}

float
sub_f(
    float arg1,
    float arg2)
{
    return arg1 - arg2;
}

float
mult_f(
    float arg1,
    float arg2)
{
    return arg1 * arg2;
}

float
div_f(
    float arg1,
    float arg2)
{
    return arg1 / arg2;
}

int
allclose_f (
    float rtol,
    float atol,
    int n_dims,
    int *shape,
    int elem_size,
    int *stride1,
    int offset1,
    char *dat1,
    int *stride2,
    int offset2,
    char *dat2)
{
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index1 = offset1;
    int f_index2 = offset2;
    for (int i = 0; i < numel; ++i) {
        float elem1 = *(float *) (dat1 + f_index1);
        float elem2 = *(float *) (dat2 + f_index2);
        if (fabsf(elem1 - elem2) > (atol + rtol * fabsf(elem2))) {
            return 0;
        }
        for (int dim = n_dims - 1; dim >= 0; --dim) {
            index[dim] += 1;
            f_index1 += stride1[dim];
            f_index2 += stride2[dim];
            if (index[dim] == shape[dim]) {
                index[dim] = 0;
                f_index1 -= stride1[dim] * shape[dim];
                f_index2 -= stride2[dim] * shape[dim];
            } else {
                break;
            }
        }
    }
    free(index);
    return 1;
}
