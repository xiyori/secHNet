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
total_elems(
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
copy(
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
    INIT_INDEX(n_dims, dat_from)
    for (size_t i = 0; i < numel; ++i) {
        memcpy(dat_to, dat_from, elem_size);
        dat_to += elem_size;
        ADVANCE_INDEX(n_dims, dat_from)
    }
    DESTROY_INDEX()
}

int
equal(
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
    INIT_INDEX2(n_dims, dat1, dat2)
    for (size_t i = 0; i < numel; ++i) {
        if (memcmp(dat1, dat2, elem_size)) {
            return 0;
        }
        ADVANCE_INDEX2(n_dims, dat1, dat2)
    }
    DESTROY_INDEX()
    return 1;
}

int
validate_tensor_index(
    size_t dim,
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    char *dat,
    long long *out)
{
    INIT_INDEX(n_dims, dat)
    for (size_t i = 0; i < numel; ++i) {
        long long elem = *(long long *) dat;
        long long norm_elem = (elem < 0) ? dim + elem : elem;
        if (norm_elem < 0 || norm_elem >= dim) {
            *out = elem;
            return 0;
        }
        ADVANCE_INDEX(n_dims, dat)
    }
    DESTROY_INDEX()
    return 1;
}

TENSOR_INDEX(tensor_index, from, to)
TENSOR_INDEX(tensor_index_assign, to, from)
