#include "integral.h"

#define SCALAR_INT_DIV_INT(a, b) \
    a / b - (a % b != 0 && ((a % b < 0) != (b < 0)))
#define SCALAR_INT_DIV_FLOAT32(a, b) \
    floorf(a / b)
#define SCALAR_INT_DIV_FLOAT64(a, b) \
    floor(a / b)

#define SCALAR_MOD_INT(a, b) \
    (b < 0) ? \
    (((-a) % (-b) < 0) ? -((-a) % (-b) + (-b)) : -((-a) % (-b))) : \
    ((a % b < 0) ? a % b + b : a % b)
#define SCALAR_MOD_FLOAT32(a, b) \
    a - b * floorf(a / b)
#define SCALAR_MOD_FLOAT64(a, b) \
    a - b * floor(a / b)


FUNC_INT_FLOAT32_64(ELEMENTWISE, int_div, SCALAR_INT_DIV)
FUNC_INT_FLOAT32_64(ELEMENTWISE, mod,     SCALAR_MOD)
