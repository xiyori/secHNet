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
    int offset,
    int elem_size,
    char *dat_from,
    char *dat_to)
{
    for (int dim = n_dims - 1; dim >= -1; --dim) {
        if (dim == -1 || stride[dim] != elem_size) {
            n_dims = dim + 1;
            break;
        }
        elem_size *= shape[dim];
    }
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index = offset;
    for (int i = 0; i < numel; ++i) {
        memcpy(
            dat_to + i * elem_size,
            dat_from + f_index,
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

int
equal (
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
    for (int dim = n_dims - 1; dim >= -1; --dim) {
        if (dim == -1 || stride1[dim] != elem_size ||
                         stride2[dim] != elem_size) {
            n_dims = dim + 1;
            break;
        }
        elem_size *= shape[dim];
    }
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index1 = offset1;
    int f_index2 = offset2;
    for (int i = 0; i < numel; ++i) {
        if (memcmp(dat1 + f_index1, dat2 + f_index2, elem_size)) {
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
