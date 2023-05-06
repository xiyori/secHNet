#ifndef TEMPLATE_MATMUL_H
#define TEMPLATE_MATMUL_H

#include "common.h"

#define MATMUL_PROTO(ignored) \
void \
tensor_matmul( \
    size_t m, \
    size_t n, \
    size_t k, \
    int n_dims, \
    size_t *shape, \
    dtype_t dtype, \
    CBLAS_TRANSPOSE trans1, \
    long long *stride1, \
    size_t offset1, \
    char *dat_from1, \
    CBLAS_TRANSPOSE trans2, \
    long long *stride2, \
    size_t offset2, \
    char *dat_from2, \
    char * __restrict dat_to)

#define MATMUL(dtype, ignored, gemm_function) \
void \
matmul_##dtype( \
    size_t m, \
    size_t n, \
    size_t k, \
    int n_dims, \
    size_t *shape, \
    CBLAS_TRANSPOSE trans1, \
    long long *stride1, \
    size_t offset1, \
    char *dat_from1, \
    CBLAS_TRANSPOSE trans2, \
    long long *stride2, \
    size_t offset2, \
    char *dat_from2, \
    char * __restrict dat_to) \
{ \
    int start_gemm_dim = n_dims - 2; \
    size_t last_dim = m * n * sizeof(dtype##_t); \
    INIT_INDEX2(start_gemm_dim, dat_from1, dat_from2) \
    for (size_t i = 0; i < numel; ++i) { \
        gemm_function( \
            CblasRowMajor, \
            trans1, \
            trans2, \
            m, \
            n, \
            k, \
            1, \
            (dtype##_t *) dat_from1, \
            stride1[n_dims - 2] / sizeof(dtype##_t), \
            (dtype##_t *) dat_from2, \
            stride2[n_dims - 2] / sizeof(dtype##_t), \
            0, \
            (dtype##_t *) dat_to, \
            n \
        ); \
        dat_to += last_dim; \
        ADVANCE_INDEX2(start_gemm_dim, dat_from1, dat_from2) \
    } \
    DESTROY_INDEX() \
}

#define MATMUL_CASE(dtype, ignored) \
case dtype##_d: \
    matmul_##dtype( \
        m, \
        n, \
        k, \
        n_dims, \
        shape, \
        trans1, \
        stride1, \
        offset1, \
        dat_from1, \
        trans2, \
        stride2, \
        offset2, \
        dat_from2, \
        dat_to \
    ); \
    break;

#endif
