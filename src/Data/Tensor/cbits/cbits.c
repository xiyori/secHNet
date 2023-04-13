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
    int *contiguous_stride,
    int offset,
    int elem_size,
    char *datFrom,
    char *datTo)
{
    int last_dim = 1;
    for (int dim = n_dims - 1; dim >= -1; --dim) {
        if (dim == -1 || stride[dim] != contiguous_stride[dim]) {
            n_dims = dim + 1;
            elem_size *= last_dim;
            break;
        }
        last_dim *= shape[dim];
    }
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
