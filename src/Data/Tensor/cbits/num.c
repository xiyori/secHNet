#include "num.h"

#define SCALAR_ABS(arg) (arg < 0) ? -arg : arg
#define SCALAR_SIGN(arg) (0 < arg) - (arg < 0)
#define SCALAR_ADD(a, b) a + b
#define SCALAR_SUB(a, b) a - b
#define SCALAR_MULT(a, b) a * b


FUNC(MAP, abs,  SCALAR_ABS)
FUNC(MAP, sign, SCALAR_SIGN)
FUNC(MAP, neg,  -)

FUNC(ELEMENTWISE, add,  SCALAR_ADD)
FUNC(ELEMENTWISE, sub,  SCALAR_SUB)
FUNC(ELEMENTWISE, mult, SCALAR_MULT)
