#ifndef NUM_H
#define NUM_H

#include "template/map.h"
#include "template/elementwise.h"

MAP_PROTO(abs);
MAP_PROTO(sign);
MAP_PROTO(neg);

ELEMENTWISE_PROTO(add);
ELEMENTWISE_PROTO(sub);
ELEMENTWISE_PROTO(mult);

#endif
