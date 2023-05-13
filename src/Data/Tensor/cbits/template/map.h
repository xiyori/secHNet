#ifndef TEMPLATE_MAP_H
#define TEMPLATE_MAP_H

#include "common.h"
#include "vectorized.h"

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
    VECTORIZED_LOOP( \
        dtype, \
        dtype_to, \
        function, \
        stride[dim] != last_dim, \
        INIT_INDEX(vec_n_dims, dat_from), \
        ADVANCE_INDEX(vec_n_dims, dat_from), \
        char *current_dat_from = dat_from, \
        current_dat_from += SIZEOF_AVX512, \
        ((dtype##_t *) current_dat_from)[k] \
    ) \
    INIT_INDEX(n_dims, dat_from) \
    for (size_t i = 0; i < numel; ++i) { \
        *(dtype_to##_t *) dat_to = function(*(dtype##_t *) dat_from); \
        dat_to += sizeof(dtype_to##_t); \
        ADVANCE_INDEX(n_dims, dat_from) \
    } \
    DESTROY_INDEX() \
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
