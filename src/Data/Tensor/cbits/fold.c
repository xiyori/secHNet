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
