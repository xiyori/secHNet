#ifndef CBITS_H
#define CBITS_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>


// typedef ? bool;
// typedef char               int8_t;
// typedef unsigned char      uint8_t;
// typedef short              int16_t;
// typedef unsigned short     uint16_t;
// typedef int                int32_t;
// typedef unsigned int       uint32_t;
// typedef long long          int64_t;
// typedef unsigned long long uint64_t;
// typedef ? float16;
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


int
flatten_index(
    int n_dims,
    int *stride,
    int offset,
    int *index);

char *
get_elem(
    int n_dims,
    int *stride,
    int offset,
    dtype_t dtype,
    char *dat,
    int *index);

int
total_elems (
    int n_dims,
    int *shape);

void
copy (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    int elem_size,
    char *dat_from,
    char *dat_to);

int
equal (
    int n_dims,
    int *shape,
    int elem_size,
    int *stride1,
    int offset1,
    char *dat1,
    int *stride2,
    int offset2,
    char *dat2);

void
map (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    int elem_size,
    char *dat_from,
    char *dat_to,
    void (*f)(char *, char *));

void
elementwise (
    int n_dims,
    int *shape,
    int elem_size,
    int *stride1,
    int offset1,
    char *dat_from1,
    int *stride2,
    int offset2,
    char *dat_from2,
    char *dat_to,
    void (*f)(char *, char *, char *));

#endif
