#include "boolean.h"

#define SCALAR_EQUAL(arg1, arg2) arg1 == arg2
#define SCALAR_NOT_EQUAL(arg1, arg2) arg1 != arg2
#define SCALAR_GREATER(arg1, arg2) arg1 > arg2
#define SCALAR_LESS(arg1, arg2) arg1 < arg2
#define SCALAR_GEQ(arg1, arg2) arg1 >= arg2
#define SCALAR_LEQ(arg1, arg2) arg1 <= arg2

#define SCALAR_NOT(arg) !arg


FUNC_MATH(ALLCLOSE, fabs)

FUNC_GENERIC(ELEMENTWISE, equal, cbool, SCALAR_EQUAL)
FUNC_GENERIC(ELEMENTWISE, not_equal, cbool, SCALAR_NOT_EQUAL)
FUNC_GENERIC(ELEMENTWISE, greater, cbool, SCALAR_GREATER)
FUNC_GENERIC(ELEMENTWISE, less, cbool, SCALAR_LESS)
FUNC_GENERIC(ELEMENTWISE, geq, cbool, SCALAR_GEQ)
FUNC_GENERIC(ELEMENTWISE, leq, cbool, SCALAR_LEQ)

MAP(cbool, not, SCALAR_NOT)
FUNC_WRAPPER(FOR_BOOL, MAP, not)
