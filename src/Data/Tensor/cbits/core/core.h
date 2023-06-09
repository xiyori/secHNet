#ifndef CORE_H
#define CORE_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "dtype.h"
#include "index.h"
#include "tensor_index.h"


size_t
flatten_index(
    int n_dims,
    long long *stride,
    size_t offset,
    size_t *index);

char *
get_elem(
    int n_dims,
    long long *stride,
    size_t offset,
    dtype_t dtype,
    char *dat,
    size_t *index);

size_t
total_elems(
    int n_dims,
    size_t *shape);

void
copy(
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    size_t elem_size,
    char *dat_from,
    char * __restrict dat_to);

int
equal(
    int n_dims,
    size_t *shape,
    size_t elem_size,
    long long *stride1,
    size_t offset1,
    char *dat1,
    long long *stride2,
    size_t offset2,
    char *dat2);

int
validate_tensor_index(
    size_t dim,
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    char *dat,
    long long *out);

TENSOR_INDEX_PROTO(tensor_index);
TENSOR_INDEX_PROTO(tensor_index_assign);

#endif
