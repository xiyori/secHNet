#ifndef TEMPLATE_VECTORIZED_H
#define TEMPLATE_VECTORIZED_H

#define VECTORIZED_LOOP( \
    dtype, dtype_to, function, \
    contiguous_condition, \
    init_index, \
    advance_index, \
    init_vector_index, \
    advance_vector_index, \
    ... \
) \
size_t last_dim = sizeof(dtype##_t); \
size_t last_dim_to = sizeof(dtype_to##_t); \
int vec_n_dims; \
for (int dim = n_dims - 1; dim >= -1; --dim) { \
    if (dim == -1 || contiguous_condition) { \
        vec_n_dims = dim + 1; \
        break; \
    } \
    last_dim *= shape[dim]; \
    last_dim_to *= shape[dim]; \
} \
if (last_dim >= SIZEOF_AVX512) { \
    init_index; \
    for (size_t i = 0; i < numel; ++i) { \
        char *current_dat_to = dat_to; \
        init_vector_index; \
        for (size_t j = 0; j < last_dim / SIZEOF_AVX512; ++j) { \
            for (size_t k = 0; k < VECTORIZED_SIZE(dtype); ++k) { \
                ((dtype_to##_t *) current_dat_to)[k] = function(__VA_ARGS__); \
            } \
            current_dat_to += VECTORIZED_SIZE(dtype) * sizeof(dtype_to##_t); \
            advance_vector_index; \
        } \
        for (size_t k = 0; k < (last_dim % SIZEOF_AVX512) / \
                                sizeof(dtype##_t); ++k) { \
            ((dtype_to##_t *) current_dat_to)[k] = function(__VA_ARGS__); \
        } \
        dat_to += last_dim_to; \
        advance_index; \
    } \
    DESTROY_INDEX() \
    return; \
}

#endif
