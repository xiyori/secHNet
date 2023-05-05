#ifndef FOLD_H
#define FOLD_H

#include <math.h>

#include "template/fold.h"

FOLD_PROTO(min);
FOLD_PROTO(max);
FOLD_PROTO(sum);

void
sum_along_dims(
    int start_sum_dim,
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    dtype_t dtype,
    size_t elem_size,
    char *dat_from,
    char * __restrict dat_to);

#endif
