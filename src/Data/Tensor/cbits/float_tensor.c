#include <stdlib.h>
#include "cbits.h"

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
    char *data0 = dat + offset;
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index = 0;
    float sum = 0, c = 0;
    for (int i = 0; i < numel; ++i) {
        float y = *(float *) (data0 + f_index) - c;
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
