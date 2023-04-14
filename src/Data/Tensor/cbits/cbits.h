int flatten_index(
    int n_dims,
    int *stride,
    int offset,
    int *index
);

char *
get_elem(
    int n_dims,
    int *stride,
    int offset,
    char *dat,
    int *index
);

// void unravel_index(
//     int n_dims,
//     int *stride,
//     int offset,
//     int f_index,
//     int *index
// );

int
total_elems (
    int n_dims,
    int *shape
);

void
copy (
    int n_dims,
    int *shape,
    int *stride,
    int *contiguous_stride,
    int offset,
    int elem_size,
    char *dat_from,
    char *dat_to
);
