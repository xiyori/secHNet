#ifndef CORE_DTYPE_H
#define CORE_DTYPE_H

#include <stdint.h>
#include <stdbool.h>


// -- Vectorized size
// -- ---------------

#define SIZEOF_AVX512 32

#define BOOL_VECTORIZED_SIZE    SIZEOF_AVX512
#define INT8_VECTORIZED_SIZE    SIZEOF_AVX512
#define UINT8_VECTORIZED_SIZE   SIZEOF_AVX512
#define INT16_VECTORIZED_SIZE   SIZEOF_AVX512 / 2
#define UINT16_VECTORIZED_SIZE  SIZEOF_AVX512 / 2
#define INT32_VECTORIZED_SIZE   SIZEOF_AVX512 / 4
#define UINT32_VECTORIZED_SIZE  SIZEOF_AVX512 / 4
#define INT64_VECTORIZED_SIZE   SIZEOF_AVX512 / 8
#define UINT64_VECTORIZED_SIZE  SIZEOF_AVX512 / 8
#define FLOAT32_VECTORIZED_SIZE SIZEOF_AVX512 / 4
#define FLOAT64_VECTORIZED_SIZE SIZEOF_AVX512 / 8

#define VECTORIZED_SIZE(dtype) \
VECTORIZED_SIZE_(DTYPE_UPPER(dtype))
#define VECTORIZED_SIZE_(uppercase_dtype) \
CONCATENATE(uppercase_dtype, _VECTORIZED_SIZE)


// -- Limits
// -- ------

#define BOOL_MIN  0
#define BOOL_MAX  1

#define UINT8_MIN  0
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

#define DTYPE_UPPER_cbool BOOL
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


typedef bool   cbool_t;
// typedef ?   float16_t;
typedef float  float32_t;
typedef double float64_t;

typedef enum {
    cbool_d,
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

#endif
