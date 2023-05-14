#ifndef BOOLEAN_H
#define BOOLEAN_H

#include "template/map.h"
#include "template/elementwise.h"
#include "template/fold.h"
#include "template/allclose.h"

ALLCLOSE_PROTO(_);

ELEMENTWISE_PROTO(equal);
ELEMENTWISE_PROTO(not_equal);
ELEMENTWISE_PROTO(greater);
ELEMENTWISE_PROTO(less);
ELEMENTWISE_PROTO(geq);
ELEMENTWISE_PROTO(leq);

MAP_PROTO(not);

FOLD_PROTO(any);
FOLD_PROTO(all);

#endif
