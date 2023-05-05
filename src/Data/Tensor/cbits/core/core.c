#include "core.h"

// #include <stdio.h>


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

void
tensor_index(
    int start_index_dim,
    int n_indices,
    int index_n_dims,
    size_t *index_shape,
    long long **index_strides,
    size_t *index_offsets,
    char **index_dat,
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    size_t elem_size,
    char *dat_from,
    char * __restrict dat_to)
{
    int copy_n_dims = n_dims - (start_index_dim + n_indices);
    size_t *copy_shape = shape + start_index_dim + n_indices;
    long long *copy_stride = stride + start_index_dim + n_indices;
    size_t last_dim = elem_size;
    for (int dim = start_index_dim + n_indices; dim < n_dims; ++dim) {
        last_dim *= shape[dim];
    }
    size_t *index_index = calloc(index_n_dims, sizeof(size_t));
    for (int dim = 0; dim < n_indices; ++dim) {
        index_dat[dim] += index_offsets[dim];
    }
    size_t index_numel = total_elems(index_n_dims, index_shape);
    size_t last_dim_copy = index_numel * last_dim;
    INIT_INDEX(start_index_dim, dat_from)
    for (size_t index_i = 0; index_i < index_numel; ++index_i) {
        char *current_dat_to = dat_to;
        char *current_dat_from = dat_from;
        for (int dim = start_index_dim; dim < start_index_dim + n_indices; ++dim) {
            long long i = *(long long *) index_dat[dim - start_index_dim];
            current_dat_from += stride[dim] * ((i < 0) ? shape[dim] + i : i);
        }
        // printf(
        //     "%d %lu %lld %lu %lu %lu %lu\n",
        //     copy_n_dims,
        //     shape[0],
        //     stride[1],
        //     elem_size,
        //     last_dim,
        //     index_numel,
        //     current_dat_from - dat_from
        // );
        for (size_t i = 0; i < numel; ++i) {
            copy(
                copy_n_dims,
                copy_shape,
                copy_stride,
                0,
                elem_size,
                current_dat_from,
                current_dat_to
            );
            current_dat_to += last_dim_copy;
            ADVANCE_INDEX(start_index_dim, current_dat_from)
        }
        dat_to += last_dim;
        for (int dim = index_n_dims - 1; dim >= 0; --dim) {
            index_index[dim] += 1;
            for (int i = 0; i < n_indices; ++i) {
                index_dat[i] += index_strides[i][dim];
            }
            if (index_index[dim] == index_shape[dim]) {
                index_index[dim] = 0;
                for (int i = 0; i < n_indices; ++i) {
                    index_dat[i] -= index_strides[i][dim] * index_shape[dim];
                }
            } else {
                break;
            }
        }
    }
    DESTROY_INDEX()
    free(index_index);
}
