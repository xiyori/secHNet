#ifndef NUM_TENSOR_H
#define NUM_TENSOR_H

#include <math.h>
#include <stdio.h>

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

#define MAP_PROTO_SEMICOLON(name) \
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

#define EYE_PROTO(ignored) \
void \
tensor_eye ( \
    size_t rows, \
    size_t columns, \
    long long k, \
    dtype_t dtype, \
    char *dat)

#define FOLD_PROTO(name) \
void \
tensor_##name ( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    dtype_t dtype, \
    char *dat, \
    char *out) \

#define FORALL_MATH(expr) \
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


MAP_PROTO(abs);
MAP_PROTO(sign);
MAP_PROTO(neg);

FORALL_MATH(MAP_PROTO_SEMICOLON)

ELEMENTWISE_PROTO(add);
ELEMENTWISE_PROTO(sub);
ELEMENTWISE_PROTO(mult);
ELEMENTWISE_PROTO(div);

ELEMENTWISE_PROTO(equal);
ELEMENTWISE_PROTO(not_equal);
ELEMENTWISE_PROTO(greater);
ELEMENTWISE_PROTO(less);
ELEMENTWISE_PROTO(geq);
ELEMENTWISE_PROTO(leq);
MAP_PROTO(not);

ALLCLOSE_PROTO(_);

EYE_PROTO(_);

FOLD_PROTO(min);
FOLD_PROTO(max);
FOLD_PROTO(sum);

#endif
