#include "num_tensor.h"

// #include <stdio.h>

#define FORALL_DTYPES(expr, ...) \
expr(int8, __VA_ARGS__) \
expr(uint8, __VA_ARGS__) \
expr(int16, __VA_ARGS__) \
expr(uint16, __VA_ARGS__) \
expr(int32, __VA_ARGS__) \
expr(uint32, __VA_ARGS__) \
expr(int64, __VA_ARGS__) \
expr(uint64, __VA_ARGS__) \
expr(float32, __VA_ARGS__) \
expr(float64, __VA_ARGS__)

#define FORALL_FLOAT_DTYPES(expr, ...) \
expr(float32, __VA_ARGS__) \
expr(float64, __VA_ARGS__)

#define ANY_FLOAT_FUNC(macro, name) \
macro(float32, name, name##f) \
macro(float64, name, name) \
\
macro##_PROTO(name) \
{ \
    switch(dtype) { \
        FORALL_FLOAT_DTYPES(macro##_CASE, name) \
    } \
}

#define MAP(dtype, name, function) \
void \
name##_##dtype ( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    char *dat_from, \
    char * __restrict dat_to) \
{ \
    size_t last_dim = sizeof(dtype##_t); \
    int vec_n_dims; \
    for (int dim = n_dims - 1; dim >= -1; --dim) { \
        if (dim == -1 || stride[dim] != last_dim) { \
            vec_n_dims = dim + 1; \
            break; \
        } \
        last_dim *= shape[dim]; \
    } \
    /* Vectorize operations for contiguous tensors */ \
    if (last_dim >= SIZEOF_AVX512) { \
        size_t *index = calloc(vec_n_dims, sizeof(size_t)); \
        size_t numel = total_elems(vec_n_dims, shape); \
        size_t f_index = offset; \
        for (size_t i = 0; i < numel; ++i) { \
            char *current_dat_to = dat_to + i * last_dim; \
            char *current_dat_from = dat_from + f_index; \
            for (size_t j = 0; j < last_dim / SIZEOF_AVX512; ++j) { \
                for (size_t k = 0; k < VECTORIZED_SIZE_##dtype; ++k) { \
                    ((dtype##_t *) current_dat_to)[k] = function( \
                        ((dtype##_t *) current_dat_from)[k] \
                    ); \
                } \
                current_dat_to += SIZEOF_AVX512; \
                current_dat_from += SIZEOF_AVX512; \
            } \
            for (size_t k = 0; k < (last_dim % SIZEOF_AVX512) / \
                                sizeof(dtype##_t); ++k) { \
                ((dtype##_t *) current_dat_to)[k] = function( \
                    ((dtype##_t *) current_dat_from)[k] \
                ); \
                /* printf("%ld %ld %f %f\n", current_dat_from - dat_from + k * sizeof(dtype##_t), \
                       (current_dat_to - dat_to) / sizeof(dtype##_t) + k, \
                       *(float *) (current_dat_to + k * sizeof(dtype##_t)), \
                       *(float *) (current_dat_from + k * sizeof(dtype##_t))); */ \
            } \
            for (size_t j = 0; j < last_dim / sizeof(dtype##_t); ++j) { \
                /* printf("%ld %ld %f %f\n", f_index + j * sizeof(dtype##_t), \
                       i * last_dim / sizeof(dtype##_t) + j, \
                       *(float *) (dat_to + i * last_dim + j * sizeof(dtype##_t)), \
                       *(float *) (dat_from + f_index + j * sizeof(dtype##_t))); */ \
            } \
            ADVANCE_INDEX(vec_n_dims) \
        } \
        free(index); \
    } else { \
        size_t *index = calloc(n_dims, sizeof(size_t)); \
        size_t numel = total_elems(n_dims, shape); \
        size_t f_index = offset; \
        for (size_t i = 0; i < numel; ++i) { \
            *(dtype##_t *) (dat_to + i * sizeof(dtype##_t)) = function( \
                *(dtype##_t *) (dat_from + f_index) \
            ); \
            ADVANCE_INDEX(n_dims) \
        } \
        free(index); \
    } \
}

#define MAP_ID(dtype, name) \
void \
name##_##dtype ( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    char *dat_from, \
    char * __restrict dat_to) \
{ \
    copy( \
        n_dims, \
        shape, \
        stride, \
        offset, \
        sizeof(dtype##_t), \
        dat_from, \
        dat_to \
    ); \
}

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

#define MAP_WRAPPER(name) \
MAP_PROTO(name) \
{ \
    switch(dtype) { \
        FORALL_DTYPES(MAP_CASE, name) \
    } \
}

#define MAP_FUNC(name, function) \
FORALL_DTYPES(MAP, name, function) \
\
MAP_WRAPPER(name)

#define MAP_FLOAT_FUNC(name) ANY_FLOAT_FUNC(MAP, name)

#define ELEMENTWISE(dtype, name, operator) \
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
    int vec_n_dims; \
    for (int dim = n_dims - 1; dim >= -1; --dim) { \
        if (dim == -1 || stride1[dim] != last_dim || \
                         stride2[dim] != last_dim) { \
            vec_n_dims = dim + 1; \
            break; \
        } \
        last_dim *= shape[dim]; \
    } \
    /* Vectorize operations for contiguous tensors */ \
    if (last_dim >= SIZEOF_AVX512) { \
        size_t *index = calloc(vec_n_dims, sizeof(size_t)); \
        size_t numel = total_elems(vec_n_dims, shape); \
        size_t f_index1 = offset1; \
        size_t f_index2 = offset2; \
        for (size_t i = 0; i < numel; ++i) { \
            char *current_dat_to = dat_to + i * last_dim; \
            char *current_dat_from1 = dat_from1 + f_index1; \
            char *current_dat_from2 = dat_from2 + f_index2; \
            for (size_t j = 0; j < last_dim / SIZEOF_AVX512; ++j) { \
                for (size_t k = 0; k < VECTORIZED_SIZE_##dtype; ++k) { \
                    ((dtype##_t *) current_dat_to)[k] = operator( \
                        ((dtype##_t *) current_dat_from1)[k],  \
                        ((dtype##_t *) current_dat_from2)[k] \
                    ); \
                } \
                current_dat_to += SIZEOF_AVX512; \
                current_dat_from1 += SIZEOF_AVX512; \
                current_dat_from2 += SIZEOF_AVX512; \
            } \
            for (size_t k = 0; k < (last_dim % SIZEOF_AVX512) / \
                                sizeof(dtype##_t); ++k) { \
                ((dtype##_t *) current_dat_to)[k] = operator( \
                    ((dtype##_t *) current_dat_from1)[k],  \
                    ((dtype##_t *) current_dat_from2)[k] \
                ); \
            } \
            ADVANCE_INDEX2(vec_n_dims) \
        } \
        free(index); \
    } else { \
        size_t *index = calloc(n_dims, sizeof(size_t)); \
        size_t numel = total_elems(n_dims, shape); \
        size_t f_index1 = offset1; \
        size_t f_index2 = offset2; \
        for (size_t i = 0; i < numel; ++i) { \
            *(dtype##_t *) (dat_to + i * sizeof(dtype##_t)) = operator( \
                *(dtype##_t *) (dat_from1 + f_index1), \
                *(dtype##_t *) (dat_from2 + f_index2) \
            ); \
            ADVANCE_INDEX2(n_dims) \
        } \
        free(index); \
    } \
}

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

#define ELEMENTWISE_FUNC(name, operator) \
FORALL_DTYPES(ELEMENTWISE, name, operator) \
\
ELEMENTWISE_PROTO(name) \
{ \
    switch(dtype) { \
        FORALL_DTYPES(ELEMENTWISE_CASE, name) \
    } \
}

#define ALLCLOSE(dtype, ignored, abs_function) \
int \
allclose_##dtype ( \
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
    size_t *index = calloc(n_dims, sizeof(size_t)); \
    size_t numel = total_elems(n_dims, shape); \
    size_t f_index1 = offset1; \
    size_t f_index2 = offset2; \
    for (size_t i = 0; i < numel; ++i) { \
        dtype##_t elem1 = *(dtype##_t *) (dat1 + f_index1); \
        dtype##_t elem2 = *(dtype##_t *) (dat2 + f_index2); \
        if (abs_function(elem1 - elem2) > (atol + rtol * abs_function(elem2))) { \
            return 0; \
        } \
        ADVANCE_INDEX2(n_dims) \
    } \
    free(index); \
    return 1; \
}

#define ALLCLOSE_CASE(dtype, ignored) \
case dtype##_d: \
    return allclose_##dtype( \
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
    );

#define ALLCLOSE_FUNC ANY_FLOAT_FUNC(ALLCLOSE, fabs)

#define SCALAR_SIGN(arg) (0 < arg) - (arg < 0)

#define SCALAR_ADD(arg1, arg2) arg1 + arg2
#define SCALAR_SUB(arg1, arg2) arg1 - arg2
#define SCALAR_MULT(arg1, arg2) arg1 * arg2
#define SCALAR_DIV(arg1, arg2) arg1 / arg2


MAP(int8, abs, abs)
MAP_ID(uint8, abs)
MAP(int16, abs, abs)
MAP_ID(uint16, abs)
MAP(int32, abs, abs)
MAP_ID(uint32, abs)
MAP(int64, abs, llabs)
MAP_ID(uint64, abs)
MAP(float32, abs, fabsf)
MAP(float64, abs, fabs)

MAP_WRAPPER(abs)
MAP_FUNC(sign, SCALAR_SIGN)
MAP_FUNC(neg, -)

FORALL_FLOATING(MAP_FLOAT_FUNC)

ELEMENTWISE_FUNC(add, SCALAR_ADD)
ELEMENTWISE_FUNC(sub, SCALAR_SUB)
ELEMENTWISE_FUNC(mult, SCALAR_MULT)
ELEMENTWISE_FUNC(div, SCALAR_DIV)

ALLCLOSE_FUNC

void
eye_f (
    int rows,
    int columns,
    int k,
    float *dat)
{
    int min_dim = (rows < columns) ? rows : columns;
    int columnShift = (k < 0) ? -k : 0;
    int rowShift = (k > 0) ? k : 0;
    for (int i = 0; i < min_dim - columnShift - rowShift; ++i) {
        dat[(i + columnShift) * columns + i + rowShift] = 1;
    }
}

float
sum_f (
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    char *dat)
{
    size_t *index = calloc(n_dims, sizeof(size_t));
    int numel = total_elems(n_dims, shape);
    int f_index = offset;
    float sum = 0, c = 0;
    for (int i = 0; i < numel; ++i) {
        float y = *(float *) (dat + f_index) - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        ADVANCE_INDEX(n_dims)
    }
    free(index);
    return sum;
}
