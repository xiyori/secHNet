#include "fold.h"

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


FUNC_INT_FLOAT(FOLD, min, MIN)
FUNC_INT_FLOAT(FOLD, max, MAX)
FUNC_INT_FLOAT(FOLD, sum, SUM)

void
sum_along_dims(
    int start_sum_dim,
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    dtype_t dtype,
    size_t elem_size,
    char *dat_from,
    char * __restrict dat_to)
{
    int sum_n_dims = n_dims - start_sum_dim;
    size_t *sum_shape = shape + start_sum_dim;
    long long *sum_stride = stride + start_sum_dim;
    INIT_INDEX(start_sum_dim, dat_from)
    for (size_t i = 0; i < numel; ++i) {
        tensor_sum(
            sum_n_dims,
            sum_shape,
            sum_stride,
            0,
            dtype,
            dat_from,
            dat_to
        );
        dat_to += elem_size;
        ADVANCE_INDEX(start_sum_dim, dat_from)
    }
    DESTROY_INDEX()
}
