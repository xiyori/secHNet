#ifndef NUM_TENSOR_H
#define NUM_TENSOR_H

#include <math.h>

#include "cbits.h"

#define MAP_PROTO(name) \
void \
tensor_##name ( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    dtype_t dtype, \
    char *dat_from, \
    char *dat_to)

#define MAP_PROTO_COMPLETE(name) \
MAP_PROTO(name);

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

#define FORALL_FLOATING(expr) \
expr(exp) \
expr(log) \
expr(sin) \
expr(cos) \
expr(asin) \
expr(acos) \
expr(atan) \
expr(sinh) \
expr(cosh) \
expr(asinh) \
expr(acosh) \
expr(atanh)

#define ALLCLOSE_PROTO(ignored) \
int \
tensor_allclose ( \
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


MAP_PROTO(abs);
MAP_PROTO(sign);
MAP_PROTO(neg);

FORALL_FLOATING(MAP_PROTO_COMPLETE)

ELEMENTWISE_PROTO(add);
ELEMENTWISE_PROTO(sub);
ELEMENTWISE_PROTO(mult);
ELEMENTWISE_PROTO(div);

ALLCLOSE_PROTO(_);

#endif
