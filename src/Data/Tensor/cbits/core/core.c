#include "core.h"


size_t
flatten_index(
    int n_dims,
    long long *stride,
    size_t offset,
    size_t *index)
{
    size_t f_index = offset;
    for (int dim = 0; dim < n_dims; ++dim) {
        f_index += stride[dim] * index[dim];
    }
    return f_index;
}

char *
get_elem(
    int n_dims,
    long long *stride,
    size_t offset,
    dtype_t dtype,
    char *dat,
    size_t *index)
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
//     long long *stride,
//     size_t offset,
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

size_t
total_elems (
    int n_dims,
    size_t *shape)
{
    size_t numel = 1;
    for (int dim = 0; dim < n_dims; ++dim) {
        numel *= shape[dim];
    }
    return numel;
}

void
copy (
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    size_t elem_size,
    char *dat_from,
    char * __restrict dat_to)
{
    for (int dim = n_dims - 1; dim >= -1; --dim) {
        if (dim == -1 || stride[dim] != elem_size) {
            n_dims = dim + 1;
            break;
        }
        elem_size *= shape[dim];
    }
    INIT_INDEX(n_dims)
    for (size_t i = 0; i < numel; ++i) {
        memcpy(
            dat_to + i * elem_size,
            dat_from + f_index,
            elem_size
        );
        ADVANCE_INDEX(n_dims)
    }
    DESTROY_INDEX()
}

int
equal (
    int n_dims,
    size_t *shape,
    size_t elem_size,
    long long *stride1,
    size_t offset1,
    char *dat1,
    long long *stride2,
    size_t offset2,
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
    INIT_INDEX2(n_dims)
    for (size_t i = 0; i < numel; ++i) {
        if (memcmp(dat1 + f_index1, dat2 + f_index2, elem_size)) {
            return 0;
        }
        ADVANCE_INDEX2(n_dims)
    }
    DESTROY_INDEX()
    return 1;
}
