#include "num.h"

#define SCALAR_SIGN(arg) (0 < arg) - (arg < 0)
#define SCALAR_ADD(arg1, arg2) arg1 + arg2
#define SCALAR_SUB(arg1, arg2) arg1 - arg2
#define SCALAR_MULT(arg1, arg2) arg1 * arg2


MAP_ID(cbool, abs)
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

FUNC(ELEMENTWISE, add,  SCALAR_ADD)
FUNC(ELEMENTWISE, sub,  SCALAR_SUB)
FUNC(ELEMENTWISE, mult, SCALAR_MULT)
