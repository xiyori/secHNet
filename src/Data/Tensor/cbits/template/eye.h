#ifndef TEMPLATE_EYE_H
#define TEMPLATE_EYE_H

#include "common.h"

#define EYE_PROTO(ignored) \
void \
tensor_eye( \
    size_t rows, \
    size_t columns, \
    long long k, \
    dtype_t dtype, \
    char *dat)

#define EYE(dtype, ignored, ignored_) \
void \
eye_##dtype( \
    size_t rows, \
    size_t columns, \
    long long k, \
    dtype##_t *dat) \
{ \
    size_t rowShift = (k < 0) ? -k : 0; \
    size_t columnShift = (k > 0) ? k : 0; \
    size_t row_nonzero = rows - rowShift; \
    size_t column_nonzero = columns - columnShift; \
    size_t total_nonzero = (row_nonzero < column_nonzero) ? row_nonzero : column_nonzero; \
    for (size_t i = 0; i < total_nonzero; ++i) { \
        dat[(i + rowShift) * columns + i + columnShift] = 1; \
    } \
}

#define EYE_CASE(dtype, ignored) \
case dtype##_d: \
    eye_##dtype( \
        rows, \
        columns, \
        k, \
        (dtype##_t *) dat \
    ); \
    break;

#endif
