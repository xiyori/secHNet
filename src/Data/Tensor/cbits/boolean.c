#include "boolean.h"

#define SCALAR_EQUAL(a, b) a == b
#define SCALAR_NOT_EQUAL(a, b) a != b
#define SCALAR_GREATER(a, b) a > b
#define SCALAR_LESS(a, b) a < b
#define SCALAR_GEQ(a, b) a >= b
#define SCALAR_LEQ(a, b) a <= b

#define SCALAR_NOT(arg) !arg

#define ANY_INIT(dtype) \
    dtype##_t accum = 0;
#define ANY_STEP(dtype, next_value) \
    if (next_value) { \
        return 1;\
    }

#define ALL_INIT(dtype) \
    dtype##_t accum = 1;
#define ALL_STEP(dtype, next_value) \
    if (!next_value) { \
        return 0;\
    }


FUNC_MATH(ALLCLOSE, fabs)

FUNC_GENERIC(ELEMENTWISE, equal, cbool, SCALAR_EQUAL)
FUNC_GENERIC(ELEMENTWISE, not_equal, cbool, SCALAR_NOT_EQUAL)
FUNC_GENERIC(ELEMENTWISE, greater, cbool, SCALAR_GREATER)
FUNC_GENERIC(ELEMENTWISE, less, cbool, SCALAR_LESS)
FUNC_GENERIC(ELEMENTWISE, geq, cbool, SCALAR_GEQ)
FUNC_GENERIC(ELEMENTWISE, leq, cbool, SCALAR_LEQ)

MAP(cbool, not, SCALAR_NOT)
FUNC_WRAPPER(FOR_BOOL, MAP, not)

FOLD(cbool, any, ANY)
FUNC_WRAPPER(FOR_BOOL, FOLD, any)

FOLD(cbool, all, ALL)
FUNC_WRAPPER(FOR_BOOL, FOLD, all)
