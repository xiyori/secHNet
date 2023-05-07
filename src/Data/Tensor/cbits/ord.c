#include "ord.h"

#define SCALAR_RELU(a) (a > 0) ? a : 0


FUNC(MAP, relu, SCALAR_RELU)
