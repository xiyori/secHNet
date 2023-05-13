#ifndef TEMPLATE_ELEMENTWISE_H
#define TEMPLATE_ELEMENTWISE_H

#include "common.h"
#include "vectorized.h"

#define ELEMENTWISE_PROTO(name) \
void \
tensor_##name( \
    int n_dims, \
    size_t *shape, \
    dtype_t dtype, \
    long long *stride1, \
    size_t offset1, \
    char *dat_from1, \
    long long *stride2, \
    size_t offset2, \
    char *dat_from2, \
    char * __restrict dat_to)

#define ELEMENTWISE_GENERIC(dtype, name, dtype_to, operator) \
void \
name##_##dtype( \
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
    int single_elem1 = 1; \
    for (int dim = 0; dim < n_dims; ++dim) { \
        single_elem1 = single_elem1 && stride1[dim] == 0; \
    } \
    int single_elem2 = 1; \
    for (int dim = 0; dim < n_dims; ++dim) { \
        single_elem2 = single_elem2 && stride2[dim] == 0; \
    } \
    if (single_elem1) { \
        VECTORIZED_LOOP( \
            dtype, \
            dtype_to, \
            operator, \
            stride2[dim] != last_dim, \
            INIT_INDEX2(vec_n_dims, dat_from1, dat_from2), \
            ADVANCE_INDEX2(vec_n_dims, dat_from1, dat_from2), \
            char *current_dat_from2 = dat_from2, \
            current_dat_from2 += SIZEOF_AVX512, \
            *(dtype##_t *) dat_from1, \
            ((dtype##_t *) current_dat_from2)[k] \
        ) \
    } else if (single_elem2) { \
        VECTORIZED_LOOP( \
            dtype, \
            dtype_to, \
            operator, \
            stride1[dim] != last_dim, \
            INIT_INDEX2(vec_n_dims, dat_from1, dat_from2), \
            ADVANCE_INDEX2(vec_n_dims, dat_from1, dat_from2), \
            char *current_dat_from1 = dat_from1, \
            current_dat_from1 += SIZEOF_AVX512, \
            ((dtype##_t *) current_dat_from1)[k], \
            *(dtype##_t *) dat_from2 \
        ) \
    } else { \
        VECTORIZED_LOOP( \
            dtype, \
            dtype_to, \
            operator, \
            stride1[dim] != last_dim || stride2[dim] != last_dim, \
            INIT_INDEX2(vec_n_dims, dat_from1, dat_from2), \
            ADVANCE_INDEX2(vec_n_dims, dat_from1, dat_from2), \
            char *current_dat_from1 = dat_from1; \
            char *current_dat_from2 = dat_from2, \
            current_dat_from1 += SIZEOF_AVX512; \
            current_dat_from2 += SIZEOF_AVX512, \
            ((dtype##_t *) current_dat_from1)[k], \
            ((dtype##_t *) current_dat_from2)[k] \
        ) \
    } \
    INIT_INDEX2(n_dims, dat_from1, dat_from2) \
    for (size_t i = 0; i < numel; ++i) { \
        *(dtype_to##_t *) dat_to = operator( \
            *(dtype##_t *) dat_from1, \
            *(dtype##_t *) dat_from2 \
        ); \
        dat_to += sizeof(dtype_to##_t); \
        ADVANCE_INDEX2(n_dims, dat_from1, dat_from2) \
    } \
    DESTROY_INDEX() \
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
