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

void
eye_f (
    int rows,
    int columns,
    int k,
    float *dat
);

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
    int offset,
    int elem_size,
    char *datFrom,
    char *datTo
);

float
sum_f (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    char *dat
);
