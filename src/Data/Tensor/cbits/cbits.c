#include <stdlib.h>
#include <string.h>

int
flatten_index(
    int n_dims,
    int *stride,
    int offset,
    int *index)
{
    int f_index = offset;
    for (int dim = 0; dim < n_dims; ++dim) {
        f_index += stride[dim] * index[dim];
    }
    return f_index;
}

char *
get_elem(
    int n_dims,
    int *stride,
    int offset,
    char *dat,
    int *index)
{
    return dat + flatten_index(
        n_dims,
        stride,
        offset,
        index
    );
}

// void
// unravel_index(
//     int n_dims,
//     int *stride,
//     int offset,
//     int f_index,
//     int *index)
// {
//     for (int dim = 0; dim < n_dims; ++dim) {
//         if (dim == 0) {
//             index[dim] = (f_index - offset) / stride[dim];
//         } else {
//             index[dim] = (f_index - offset) % stride[dim - 1] / stride[dim];
//         }
//     }
// }

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

int
total_elems (
    int n_dims,
    int *shape)
{
    int numel = 1;
    for (int dim = 0; dim < n_dims; ++dim) {
        numel *= shape[dim];
    }
    return numel;
}

void
copy (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    int elem_size,
    char *datFrom,
    char *datTo)
{
    char *dataFrom0 = datFrom + offset;
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index = 0;
    for (int i = 0; i < numel; ++i) {
        memcpy(
            datTo + i * elem_size,
            datFrom + f_index,
            elem_size
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
