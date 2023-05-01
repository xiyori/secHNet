#include "num_tensor.h"


// -- Common
// -- ------

#define FORALL_INT_DTYPES(expr, ...) \
expr(int8, __VA_ARGS__) \
expr(uint8, __VA_ARGS__) \
expr(int16, __VA_ARGS__) \
expr(uint16, __VA_ARGS__) \
expr(int32, __VA_ARGS__) \
expr(uint32, __VA_ARGS__) \
expr(int64, __VA_ARGS__) \
expr(uint64, __VA_ARGS__)

#define FORALL_FLOAT_DTYPES(expr, ...) \
expr(float32, __VA_ARGS__) \
expr(float64, __VA_ARGS__)

#define FORALL_DTYPES(expr, ...) \
FORALL_INT_DTYPES(expr, __VA_ARGS__) \
FORALL_FLOAT_DTYPES(expr, __VA_ARGS__)

#define FUNC_WRAPPER(dtype_iterator, macro, name) \
macro##_PROTO(name) \
{ \
    switch(dtype) { \
        dtype_iterator(macro##_CASE, name) \
    } \
}

#define FUNC(macro, name, function) \
FORALL_DTYPES(macro, name, function) \
\
FUNC_WRAPPER(FORALL_DTYPES, macro, name)

#define FUNC_INT_FLOAT(macro, name, function) \
FORALL_INT_DTYPES(macro, name, function##_INT) \
FORALL_FLOAT_DTYPES(macro, name, function##_FLOAT) \
\
FUNC_WRAPPER(FORALL_DTYPES, macro, name)

#define FUNC_MATH(macro, name) \
macro(float32, name, name##f) \
macro(float64, name, name) \
\
FUNC_WRAPPER(FORALL_FLOAT_DTYPES, macro, name)


// -- Map
// -- ---

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
                for (size_t k = 0; k < VECTORIZED_SIZE(dtype); ++k) { \
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

#define MAP_FUNC_MATH(name) FUNC_MATH(MAP, name)


// -- Elementwise
// -- -----------

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
                for (size_t k = 0; k < VECTORIZED_SIZE(dtype); ++k) { \
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


// -- Allclose
// -- --------

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


// -- Eye
// -- ---

#define EYE(dtype, ignored, ignored_) \
void \
eye_##dtype ( \
    size_t rows, \
    size_t columns, \
    long long k, \
    dtype##_t *dat) \
{ \
    size_t rowShift = (k < 0) ? -k : 0; \
    size_t columnShift = (k > 0) ? k : 0; \
    size_t row_nonzero = rows - rowShift; \
    size_t column_nonzero = columns - columnShift; \
    size_t total_nonzero = (row_nonzero < column_nonzero) ? row_nonzero : column_nonzero; \
    for (size_t i = 0; i < total_nonzero; ++i) { \
        dat[(i + rowShift) * columns + i + columnShift] = 1; \
    } \
}

#define EYE_CASE(dtype, ignored) \
case dtype##_d: \
    eye_##dtype( \
        rows, \
        columns, \
        k, \
        (dtype##_t *) dat \
    ); \
    break;


// -- Fold
// -- ----

#define FOLD(dtype, name, function) \
dtype##_t \
name##_##dtype ( \
    int n_dims, \
    size_t *shape, \
    long long *stride, \
    size_t offset, \
    char *dat) \
{ \
    size_t *index = calloc(n_dims, sizeof(size_t)); \
    size_t numel = total_elems(n_dims, shape); \
    size_t f_index = offset; \
    function##_INIT(dtype) \
    for (size_t i = 0; i < numel; ++i) { \
        function##_STEP(dtype, *(dtype##_t *) (dat + f_index)) \
        ADVANCE_INDEX(n_dims) \
    } \
    free(index); \
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


// -- Map operations
// -- --------------

#define SCALAR_SIGN(arg) (0 < arg) - (arg < 0)


// -- Elementwise operations
// -- ----------------------

#define SCALAR_ADD(arg1, arg2) arg1 + arg2
#define SCALAR_SUB(arg1, arg2) arg1 - arg2
#define SCALAR_MULT(arg1, arg2) arg1 * arg2
#define SCALAR_DIV(arg1, arg2) arg1 / arg2


// -- Fold operations
// -- ---------------

#define MIN_INT_INIT(dtype) \
    dtype##_t accum = MAX_VALUE(dtype);
#define MIN_INT_STEP(dtype, next_value) \
    accum = (next_value < accum) ? next_value : accum;

#define MIN_FLOAT_INIT(dtype) \
    dtype##_t accum = INFINITY;
#define MIN_FLOAT_STEP(dtype, next_value) \
    MIN_INT_STEP(dtype, next_value)

#define MAX_INT_INIT(dtype) \
    dtype##_t accum = MIN_VALUE(dtype);
#define MAX_INT_STEP(dtype, next_value) \
    accum = (next_value > accum) ? next_value : accum;

#define MAX_FLOAT_INIT(dtype) \
    dtype##_t accum = -INFINITY;
#define MAX_FLOAT_STEP(dtype, next_value) \
    MAX_INT_STEP(dtype, next_value)

#define SUM_INT_INIT(dtype) \
    dtype##_t accum = 0;
#define SUM_INT_STEP(dtype, next_value) \
    accum += next_value;

#define SUM_FLOAT_INIT(dtype) \
    dtype##_t accum = 0, c = 0;
#define SUM_FLOAT_STEP(dtype, next_value) \
    dtype##_t y = next_value - c; \
    dtype##_t t = accum + y; \
    c = (t - accum) - y; \
    accum = t;


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

FUNC_WRAPPER(FORALL_DTYPES, MAP, abs)
FUNC(MAP, sign, SCALAR_SIGN)
FUNC(MAP, neg, -)

FORALL_MATH(MAP_FUNC_MATH)

FUNC(ELEMENTWISE, add,  SCALAR_ADD)
FUNC(ELEMENTWISE, sub,  SCALAR_SUB)
FUNC(ELEMENTWISE, mult, SCALAR_MULT)
FUNC(ELEMENTWISE, div,  SCALAR_DIV)

FUNC_MATH(ALLCLOSE, fabs)

FUNC(EYE, _, _)

FUNC_INT_FLOAT(FOLD, min, MIN)
FUNC_INT_FLOAT(FOLD, max, MAX)
FUNC_INT_FLOAT(FOLD, sum, SUM)
