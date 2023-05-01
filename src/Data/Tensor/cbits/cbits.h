#ifndef CBITS_H
#define CBITS_H

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define ADVANCE_INDEX(n_dims) \
for (int dim = n_dims - 1; dim >= 0; --dim) { \
    index[dim] += 1; \
    f_index += stride[dim]; \
    if (index[dim] == shape[dim]) { \
        index[dim] = 0; \
        f_index -= stride[dim] * shape[dim]; \
    } else { \
        break; \
    } \
}

#define ADVANCE_INDEX2(n_dims) \
for (int dim = n_dims - 1; dim >= 0; --dim) { \
    index[dim] += 1; \
    f_index1 += stride1[dim]; \
    f_index2 += stride2[dim]; \
    if (index[dim] == shape[dim]) { \
        index[dim] = 0; \
        f_index1 -= stride1[dim] * shape[dim]; \
        f_index2 -= stride2[dim] * shape[dim]; \
    } else { \
        break; \
    } \
}

#define SIZEOF_AVX512 32

#define INT8_VECTORIZED_SIZE \
SIZEOF_AVX512
#define UINT8_VECTORIZED_SIZE \
SIZEOF_AVX512
#define INT16_VECTORIZED_SIZE \
SIZEOF_AVX512 / 2
#define UINT16_VECTORIZED_SIZE \
SIZEOF_AVX512 / 2
#define INT32_VECTORIZED_SIZE \
SIZEOF_AVX512 / 4
#define UINT32_VECTORIZED_SIZE \
SIZEOF_AVX512 / 4
#define INT64_VECTORIZED_SIZE \
SIZEOF_AVX512 / 8
#define UINT64_VECTORIZED_SIZE \
SIZEOF_AVX512 / 8
#define FLOAT32_VECTORIZED_SIZE \
SIZEOF_AVX512 / 4
#define FLOAT64_VECTORIZED_SIZE \
SIZEOF_AVX512 / 8

#define VECTORIZED_SIZE(dtype) \
VECTORIZED_SIZE_(DTYPE_UPPER(dtype))
#define VECTORIZED_SIZE_(uppercase_dtype) \
CONCATENATE(uppercase_dtype, _VECTORIZED_SIZE)

#define UINT8_MIN 0
#define UINT16_MIN 0
#define UINT32_MIN 0
#define UINT64_MIN 0

#define MIN_VALUE(int_dtype) \
MIN_VALUE_(DTYPE_UPPER(int_dtype))
#define MIN_VALUE_(uppercase_dtype) \
CONCATENATE(uppercase_dtype, _MIN)

#define MAX_VALUE(int_dtype) \
MAX_VALUE_(DTYPE_UPPER(int_dtype))
#define MAX_VALUE_(uppercase_dtype) \
CONCATENATE(uppercase_dtype, _MAX)

#define CONCATENATE(a, b) a##b

#define DTYPE_UPPER(dtype) DTYPE_UPPER_##dtype

#define DTYPE_UPPER_int8 INT8
#define DTYPE_UPPER_uint8 UINT8
#define DTYPE_UPPER_int16 INT16
#define DTYPE_UPPER_uint16 UINT16
#define DTYPE_UPPER_int32 INT32
#define DTYPE_UPPER_uint32 UINT32
#define DTYPE_UPPER_int64 INT64
#define DTYPE_UPPER_uint64 UINT64
#define DTYPE_UPPER_float32 FLOAT32
#define DTYPE_UPPER_float64 FLOAT64


// typedef ? bool_t;
// typedef char               int8_t;
// typedef unsigned char      uint8_t;
// typedef short              int16_t;
// typedef unsigned short     uint16_t;
// typedef int                int32_t;
// typedef unsigned int       uint32_t;
// typedef long long          int64_t;
// typedef unsigned long long uint64_t;
// typedef ? float16_t;
typedef float              float32_t;
typedef double             float64_t;

typedef enum {
    bool_d,
    int8_d,
    uint8_d,
    int16_d,
    uint16_d,
    int32_d,
    uint32_d,
    int64_d,
    uint64_d,
    float16_d,
    float32_d,
    float64_d
} dtype_t;


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
total_elems (
    int n_dims,
    size_t *shape);

void
copy (
    int n_dims,
    size_t *shape,
    long long *stride,
    size_t offset,
    size_t elem_size,
    char *dat_from,
    char * __restrict dat_to);

int
equal (
    int n_dims,
    size_t *shape,
    size_t elem_size,
    long long *stride1,
    size_t offset1,
    char *dat1,
    long long *stride2,
    size_t offset2,
    char *dat2);

#endif
