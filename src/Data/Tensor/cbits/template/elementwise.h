#ifndef TEMPLATE_ELEMENTWISE_H
#define TEMPLATE_ELEMENTWISE_H

#include "common.h"

#define ELEMENTWISE_PROTO(name) \
void \
tensor_##name ( \
    int n_dims, \
    size_t *shape, \
    dtype_t dtype, \
    long long *stride1, \
    size_t offset1, \
    char *dat_from1, \
    long long *stride2, \
    size_t offset2, \
    char *dat_from2, \
    char *dat_to)

#define ELEMENTWISE_GENERIC(dtype, name, dtype_to, operator) \
void \
name##_##dtype ( \
    int n_dims, \
    size_t *shape, \
    long long *stride1, \
    size_t offset1, \
    char *dat_from1, \
    long long *stride2, \
    size_t offset2, \
    char *dat_from2, \
    char * __restrict dat_to) \
{ \
    size_t last_dim = sizeof(dtype##_t); \
    size_t last_dim_to = sizeof(dtype_to##_t); \
    int vec_n_dims; \
    for (int dim = n_dims - 1; dim >= -1; --dim) { \
        if (dim == -1 || stride1[dim] != last_dim || \
                         stride2[dim] != last_dim) { \
            vec_n_dims = dim + 1; \
            break; \
        } \
        last_dim *= shape[dim]; \
        last_dim_to *= shape[dim]; \
    } \
    /* Vectorize operations for contiguous tensors */ \
    if (last_dim >= SIZEOF_AVX512) { \
        INIT_INDEX2(vec_n_dims) \
        for (size_t i = 0; i < numel; ++i) { \
            char *current_dat_to = dat_to + i * last_dim_to; \
            char *current_dat_from1 = dat_from1 + f_index1; \
            char *current_dat_from2 = dat_from2 + f_index2; \
            for (size_t j = 0; j < last_dim / SIZEOF_AVX512; ++j) { \
                for (size_t k = 0; k < VECTORIZED_SIZE(dtype); ++k) { \
                    ((dtype_to##_t *) current_dat_to)[k] = operator( \
                        ((dtype##_t *) current_dat_from1)[k],  \
                        ((dtype##_t *) current_dat_from2)[k] \
                    ); \
                } \
                current_dat_to += VECTORIZED_SIZE(dtype) * sizeof(dtype_to##_t); \
                current_dat_from1 += SIZEOF_AVX512; \
                current_dat_from2 += SIZEOF_AVX512; \
            } \
            for (size_t k = 0; k < (last_dim % SIZEOF_AVX512) / \
                                   sizeof(dtype##_t); ++k) { \
                ((dtype_to##_t *) current_dat_to)[k] = operator( \
                    ((dtype##_t *) current_dat_from1)[k],  \
                    ((dtype##_t *) current_dat_from2)[k] \
                ); \
            } \
            ADVANCE_INDEX2(vec_n_dims) \
        } \
        DESTROY_INDEX() \
    } else { \
        INIT_INDEX2(n_dims) \
        for (size_t i = 0; i < numel; ++i) { \
            *(dtype_to##_t *) (dat_to + i * sizeof(dtype_to##_t)) = operator( \
                *(dtype##_t *) (dat_from1 + f_index1), \
                *(dtype##_t *) (dat_from2 + f_index2) \
            ); \
            ADVANCE_INDEX2(n_dims) \
        } \
        DESTROY_INDEX() \
    } \
}

#define ELEMENTWISE(dtype, name, operator) \
ELEMENTWISE_GENERIC(dtype, name, dtype, operator)

#define ELEMENTWISE_CASE(dtype, name) \
case dtype##_d: \
    name##_##dtype( \
        n_dims, \
        shape, \
        stride1, \
        offset1, \
        dat_from1, \
        stride2, \
        offset2, \
        dat_from2, \
        dat_to \
    ); \
    break;

#endif
