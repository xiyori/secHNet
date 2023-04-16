#include "num_tensor.h"

#define FORALL_DTYPES1(expr, arg1) \
expr(int8, arg1) \
expr(uint8, arg1) \
expr(int16, arg1) \
expr(uint16, arg1) \
expr(int32, arg1) \
expr(uint32, arg1) \
expr(int64, arg1) \
expr(uint64, arg1) \
expr(float32, arg1) \
expr(float64, arg1)

#define FORALL_DTYPES2(expr, arg1, arg2) \
expr(int8, arg1, arg2) \
expr(uint8, arg1, arg2) \
expr(int16, arg1, arg2) \
expr(uint16, arg1, arg2) \
expr(int32, arg1, arg2) \
expr(uint32, arg1, arg2) \
expr(int64, arg1, arg2) \
expr(uint64, arg1, arg2) \
expr(float32, arg1, arg2) \
expr(float64, arg1, arg2)

#define FORALL_FLOAT_DTYPES1(expr, arg1) \
expr(float32, arg1) \
expr(float64, arg1)

#define MAP(dtype, name, function) \
void \
name##_##dtype ( \
    char *arg, \
    char *out) \
{ \
   *(dtype##_t *) out = function(*(dtype##_t *) arg); \
}

#define MAP_ID(dtype, name) \
void \
name##_##dtype ( \
    char *arg, \
    char *out) \
{ \
   *(dtype##_t *) out = *(dtype##_t *) arg; \
}

#define MAP_CASE(dtype, name) \
case dtype##_d: \
    map( \
        n_dims, \
        shape, \
        stride, \
        offset, \
        sizeof(dtype##_t), \
        dat_from, \
        dat_to, \
        name##_##dtype \
    ); \
    break;

#define MAP_WRAPPER(name) \
MAP_PROTO(name) \
{ \
    switch(dtype) { \
        FORALL_DTYPES1(MAP_CASE, name) \
    } \
}

#define MAP_FUNC(name, function) \
FORALL_DTYPES2(MAP, name, function) \
\
MAP_WRAPPER(name)

#define MAP_FLOAT_FUNC(name) \
MAP(float32, name, name##f) \
MAP(float64, name, name) \
\
MAP_PROTO(name) \
{ \
    switch(dtype) { \
        FORALL_FLOAT_DTYPES1(MAP_CASE, name) \
    } \
}

#define ELEMENTWISE(dtype, name, operator) \
void \
name##_##dtype ( \
    char *arg1, \
    char *arg2, \
    char *out) \
{ \
   *(dtype##_t *) out = *(dtype##_t *) arg1 operator *(dtype##_t *) arg2; \
}

#define ELEMENTWISE_CASE(dtype, name) \
case dtype##_d: \
    elementwise( \
        n_dims, \
        shape, \
        sizeof(dtype##_t), \
        stride1, \
        offset1, \
        dat_from1, \
        stride2, \
        offset2, \
        dat_from2, \
        dat_to, \
        name##_##dtype \
    ); \
    break;

#define ELEMENTWISE_FUNC(name, operator) \
FORALL_DTYPES2(ELEMENTWISE, name, operator) \
\
ELEMENTWISE_PROTO(name) \
{ \
    switch(dtype) { \
        FORALL_DTYPES1(ELEMENTWISE_CASE, name) \
    } \
}


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

#define SIGN(arg) (0 < arg) - (arg < 0)

MAP_FUNC(sign, SIGN)

MAP_FUNC(neg, -)

FORALL_FLOATING(MAP_FLOAT_FUNC)

ELEMENTWISE_FUNC(add, +)
ELEMENTWISE_FUNC(sub, -)
ELEMENTWISE_FUNC(mult, *)
ELEMENTWISE_FUNC(div, /)


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
    int *shape,
    int *stride,
    int offset,
    char *dat)
{
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index = offset;
    float sum = 0, c = 0;
    for (int i = 0; i < numel; ++i) {
        float y = *(float *) (dat + f_index) - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
        for (int dim = n_dims - 1; dim >= 0; --dim) {
            index[dim] += 1;
            f_index += stride[dim];
            if (index[dim] == shape[dim]) {
                index[dim] = 0;
                f_index -= stride[dim] * shape[dim];
            } else {
                break;
            }
        }
    }
    free(index);
    return sum;
}

int
allclose_f (
    float rtol,
    float atol,
    int n_dims,
    int *shape,
    int elem_size,
    int *stride1,
    int offset1,
    char *dat1,
    int *stride2,
    int offset2,
    char *dat2)
{
    int *index = calloc(n_dims, sizeof(int));
    int numel = total_elems(n_dims, shape);
    int f_index1 = offset1;
    int f_index2 = offset2;
    for (int i = 0; i < numel; ++i) {
        float elem1 = *(float *) (dat1 + f_index1);
        float elem2 = *(float *) (dat2 + f_index2);
        if (fabsf(elem1 - elem2) > (atol + rtol * fabsf(elem2))) {
            return 0;
        }
        for (int dim = n_dims - 1; dim >= 0; --dim) {
            index[dim] += 1;
            f_index1 += stride1[dim];
            f_index2 += stride2[dim];
            if (index[dim] == shape[dim]) {
                index[dim] = 0;
                f_index1 -= stride1[dim] * shape[dim];
                f_index2 -= stride2[dim] * shape[dim];
            } else {
                break;
            }
        }
    }
    free(index);
    return 1;
}
