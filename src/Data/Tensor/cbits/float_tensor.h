void
eye_f (
    int rows,
    int columns,
    int k,
    float *dat
);

float
sum_f (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    char *dat
);

void
map_f (
    int n_dims,
    int *shape,
    int *stride,
    int offset,
    char *dat_from,
    char *dat_to,
    float (*f)(float)
);

void
elementwise_f (
    int n_dims,
    int *shape,
    int *stride1,
    int offset1,
    char *dat_from1,
    int *stride2,
    int offset2,
    char *dat_from2,
    char *dat_to,
    float (*f)(float, float)
);

float
add_f(
    float arg1,
    float arg2
);

float
sub_f(
    float arg1,
    float arg2
);

float
mult_f(
    float arg1,
    float arg2
);

float
div_f(
    float arg1,
    float arg2
);

float
neg_f(
    float arg
);

float
sign_f(
    float arg
);
