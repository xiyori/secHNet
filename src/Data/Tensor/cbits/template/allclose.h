#ifndef TEMPLATE_ALLCLOSE_H
#define TEMPLATE_ALLCLOSE_H

#include "common.h"

#define ALLCLOSE_PROTO(ignored) \
int \
tensor_allclose( \
    float64_t rtol, \
    float64_t atol, \
    int n_dims, \
    size_t *shape, \
    dtype_t dtype, \
    long long *stride1, \
    size_t offset1, \
    char *dat1, \
    long long *stride2, \
    size_t offset2, \
    char *dat2)

#define ALLCLOSE(dtype, ignored, abs_function) \
int \
allclose_##dtype( \
    dtype##_t rtol, \
    dtype##_t atol, \
    int n_dims, \
    size_t *shape, \
    long long *stride1, \
    size_t offset1, \
    char *dat1, \
    long long *stride2, \
    size_t offset2, \
    char *dat2) \
{ \
    INIT_INDEX2(n_dims, dat1, dat2) \
    for (size_t i = 0; i < numel; ++i) { \
        dtype##_t elem1 = *(dtype##_t *) dat1; \
        dtype##_t elem2 = *(dtype##_t *) dat2; \
        if (abs_function(elem1 - elem2) > (atol + rtol * abs_function(elem2))) { \
            return 0; \
        } \
        ADVANCE_INDEX2(n_dims, dat1, dat2) \
    } \
    DESTROY_INDEX() \
    return 1; \
}

#define ALLCLOSE_CASE(dtype, ignored) \
case dtype##_d: \
    allclose_##dtype( \
        rtol, \
        atol, \
        n_dims, \
        shape, \
        stride1, \
        offset1, \
        dat1, \
        stride2, \
        offset2, \
        dat2 \
    ); \
    break;

#endif
