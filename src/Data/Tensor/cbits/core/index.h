#ifndef CORE_INDEX_H
#define CORE_INDEX_H

#define INIT_INDEX(n_dims, dat) \
    size_t *index = calloc(n_dims, sizeof(size_t)); \
    size_t numel = total_elems(n_dims, shape); \
    dat += offset;

#define INIT_INDEX2(n_dims, dat1, dat2) \
    size_t *index = calloc(n_dims, sizeof(size_t)); \
    size_t numel = total_elems(n_dims, shape); \
    dat1 += offset1; \
    dat2 += offset2;

#define ADVANCE_INDEX(n_dims, dat) \
for (int dim = n_dims - 1; dim >= 0; --dim) { \
    index[dim] += 1; \
    dat += stride[dim]; \
    if (index[dim] == shape[dim]) { \
        index[dim] = 0; \
        dat -= stride[dim] * shape[dim]; \
    } else { \
        break; \
    } \
}

#define ADVANCE_INDEX2(n_dims, dat1, dat2) \
for (int dim = n_dims - 1; dim >= 0; --dim) { \
    index[dim] += 1; \
    dat1 += stride1[dim]; \
    dat2 += stride2[dim]; \
    if (index[dim] == shape[dim]) { \
        index[dim] = 0; \
        dat1 -= stride1[dim] * shape[dim]; \
        dat2 -= stride2[dim] * shape[dim]; \
    } else { \
        break; \
    } \
}

#define DESTROY_INDEX() free(index);

#endif
