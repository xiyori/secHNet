#ifndef FLOATING_H
#define FLOATING_H

#include "template/map.h"

#define FORALL_MATH(expr) \
expr(exp) \
expr(log) \
expr(sin) \
expr(cos) \
expr(asin) \
expr(acos) \
expr(atan) \
expr(sinh) \
expr(cosh) \
expr(asinh) \
expr(acosh) \
expr(atanh)

#define MAP_PROTO_SEMICOLON(name) \
MAP_PROTO(name);


FORALL_MATH(MAP_PROTO_SEMICOLON)

#endif
