#define TENSOR_INDEX_PROTO(name) \
void \
name( \
    int n_indices, \
    int index_n_dims, \
    size_t *index_shape, \
    long long **index_strides, \
    size_t *index_offsets, \
    char **index_dat, \
    int n_dims, \
    size_t *shape, \
    size_t elem_size, \
    long long *stride1, \
    size_t offset1, \
    char *dat_from, \
    long long *stride2, \
    size_t offset2, \
    char * __restrict dat_to)

#define TENSOR_INDEX(name, indexed_suffix, suffix) \
TENSOR_INDEX_PROTO(name) \
{ \
    size_t *indexed_shape = shape; \
    long long *indexed_stride_from = stride1; \
    long long *indexed_stride_to = stride2; \
    shape += n_indices; \
    stride1 += n_indices; \
    stride2 += n_indices; \
    n_dims -= n_indices; \
    for (int dim = n_dims - 1; dim >= -1; --dim) { \
        if (dim == -1 || stride1[dim] != elem_size || \
                         stride2[dim] != elem_size) { \
            n_dims = dim + 1; \
            break; \
        } \
        elem_size *= shape[dim]; \
    } \
    size_t *index_index = calloc(index_n_dims, sizeof(size_t)); \
    for (int dim = 0; dim < n_indices; ++dim) { \
        index_dat[dim] += index_offsets[dim]; \
    } \
    size_t index_numel = total_elems(index_n_dims, index_shape); \
    INIT_INDEX2(n_dims, dat_from, dat_to) \
    for (size_t index_i = 0; index_i < index_numel; ++index_i) { \
        char *current_dat_to = dat_to; \
        char *current_dat_from = dat_from; \
        for (int dim = 0; dim < n_indices; ++dim) { \
            long long i = *(long long *) index_dat[dim]; \
            current_dat_##indexed_suffix += \
                indexed_stride_##indexed_suffix[dim] * ((i < 0) ? indexed_shape[dim] + i : i); \
        } \
        for (size_t i = 0; i < numel; ++i) { \
            memcpy(current_dat_to, current_dat_from, elem_size); \
            ADVANCE_INDEX2(n_dims, current_dat_from, current_dat_to) \
        } \
        for (int dim = index_n_dims - 1; dim >= 0; --dim) { \
            index_index[dim] += 1; \
            dat_##suffix += indexed_stride_##suffix[dim]; \
            for (int i = 0; i < n_indices; ++i) { \
                index_dat[i] += index_strides[i][dim]; \
            } \
            if (index_index[dim] == index_shape[dim]) { \
                index_index[dim] = 0; \
                dat_##suffix -= indexed_stride_##suffix[dim] * index_shape[dim]; \
                for (int i = 0; i < n_indices; ++i) { \
                    index_dat[i] -= index_strides[i][dim] * index_shape[dim]; \
                } \
            } else { \
                break; \
            } \
        } \
    } \
    DESTROY_INDEX() \
    free(index_index); \
}
