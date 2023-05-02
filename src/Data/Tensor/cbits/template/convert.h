#ifndef TEMPLATE_CONVERT_H
#define TEMPLATE_CONVERT_H

#include "map.h"

#define CONVERT_PROTO(ignored) \
void \
tensor_astype ( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    dtype_t dtype_from, \
    char *dat_from, \
    dtype_t dtype, \
    char * __restrict dat_to)

#define SCALAR_ID(arg) arg

#define CONVERT(dtype, ignored, ignored_) \
DEFER(FUNC_GENERIC)(MAP, astype_##dtype, dtype, SCALAR_ID)

#define CONVERT_CASE(dtype, name) \
case dtype##_d: \
    tensor_astype_##dtype( \
        n_dims, \
        shape, \
        stride, \
        offset, \
        dtype_from, \
        dat_from, \
        dat_to \
    ); \
    break;

#endif
