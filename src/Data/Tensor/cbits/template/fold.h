#ifndef TEMPLATE_FOLD_H
#define TEMPLATE_FOLD_H

#include "common.h"

#define FOLD_PROTO(name) \
void \
tensor_##name( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    dtype_t dtype, \
    char *dat, \
    char *out)

#define FOLD(dtype, name, function) \
dtype##_t \
name##_##dtype( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    char *dat) \
{ \
    INIT_INDEX(n_dims, dat) \
    function##_INIT(dtype) \
    for (size_t i = 0; i < numel; ++i) { \
        function##_STEP(dtype, *(dtype##_t *) dat) \
        ADVANCE_INDEX(n_dims, dat) \
    } \
    DESTROY_INDEX() \
    return accum; \
}

#define FOLD_CASE(dtype, name) \
case dtype##_d: \
    *(dtype##_t *) out = name##_##dtype( \
        n_dims, \
        shape, \
        stride, \
        offset, \
        dat \
    ); \
    break;

#endif
