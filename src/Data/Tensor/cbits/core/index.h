#ifndef CORE_INDEX_H
#define CORE_INDEX_H

#define INIT_INDEX(n_dims) \
    size_t *index = calloc(n_dims, sizeof(size_t)); \
    size_t numel = total_elems(n_dims, shape); \
    size_t f_index = offset;

#define INIT_INDEX2(n_dims) \
    size_t *index = calloc(n_dims, sizeof(size_t)); \
    size_t numel = total_elems(n_dims, shape); \
    size_t f_index1 = offset1; \
    size_t f_index2 = offset2;

#define ADVANCE_INDEX(n_dims) \
for (int dim = n_dims - 1; dim >= 0; --dim) { \
    index[dim] += 1; \
    f_index += stride[dim]; \
    if (index[dim] == shape[dim]) { \
        index[dim] = 0; \
        f_index -= stride[dim] * shape[dim]; \
    } else { \
        break; \
    } \
}

#define ADVANCE_INDEX2(n_dims) \
for (int dim = n_dims - 1; dim >= 0; --dim) { \
    index[dim] += 1; \
    f_index1 += stride1[dim]; \
    f_index2 += stride2[dim]; \
    if (index[dim] == shape[dim]) { \
        index[dim] = 0; \
        f_index1 -= stride1[dim] * shape[dim]; \
        f_index2 -= stride2[dim] * shape[dim]; \
    } else { \
        break; \
    } \
}

#define DESTROY_INDEX() free(index);

#endif
