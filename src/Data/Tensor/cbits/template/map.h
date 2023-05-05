#ifndef TEMPLATE_MAP_H
#define TEMPLATE_MAP_H

#include "common.h"

#define MAP_PROTO(name) \
void \
tensor_##name( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    dtype_t dtype, \
    char *dat_from, \
    char * __restrict dat_to)

#define MAP_GENERIC(dtype, name, dtype_to, function) \
void \
name##_##dtype( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    char *dat_from, \
    char * __restrict dat_to) \
{ \
    size_t last_dim = sizeof(dtype##_t); \
    size_t last_dim_to = sizeof(dtype_to##_t); \
    int vec_n_dims; \
    for (int dim = n_dims - 1; dim >= -1; --dim) { \
        if (dim == -1 || stride[dim] != last_dim) { \
            vec_n_dims = dim + 1; \
            break; \
        } \
        last_dim *= shape[dim]; \
        last_dim_to *= shape[dim]; \
    } \
    /* Vectorize operations for contiguous tensors */ \
    if (last_dim >= SIZEOF_AVX512) { \
        INIT_INDEX(vec_n_dims, dat_from) \
        for (size_t i = 0; i < numel; ++i) { \
            char *current_dat_to = dat_to; \
            char *current_dat_from = dat_from; \
            for (size_t j = 0; j < last_dim / SIZEOF_AVX512; ++j) { \
                for (size_t k = 0; k < VECTORIZED_SIZE(dtype); ++k) { \
                    ((dtype_to##_t *) current_dat_to)[k] = function( \
                        ((dtype##_t *) current_dat_from)[k] \
                    ); \
                } \
                current_dat_to += VECTORIZED_SIZE(dtype) * sizeof(dtype_to##_t); \
                current_dat_from += SIZEOF_AVX512; \
            } \
            for (size_t k = 0; k < (last_dim % SIZEOF_AVX512) / \
                                   sizeof(dtype##_t); ++k) { \
                ((dtype_to##_t *) current_dat_to)[k] = function( \
                    ((dtype##_t *) current_dat_from)[k] \
                ); \
            } \
            dat_to += last_dim_to; \
            ADVANCE_INDEX(vec_n_dims, dat_from) \
        } \
        DESTROY_INDEX() \
    } else { \
        INIT_INDEX(n_dims, dat_from) \
        for (size_t i = 0; i < numel; ++i) { \
            *(dtype_to##_t *) dat_to = function(*(dtype##_t *) dat_from); \
            dat_to += sizeof(dtype_to##_t); \
            ADVANCE_INDEX(n_dims, dat_from) \
        } \
        DESTROY_INDEX() \
    } \
}

#define MAP(dtype, name, function) \
MAP_GENERIC(dtype, name, dtype, function)

#define MAP_CASE(dtype, name) \
case dtype##_d: \
    name##_##dtype( \
        n_dims, \
        shape, \
        stride, \
        offset, \
        dat_from, \
        dat_to \
    ); \
    break;

#endif
