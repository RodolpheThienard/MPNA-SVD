#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"
#define ITERMAX 300
#define ERR 0.001
void bidiagonalisation (f64 *matrix_a, f64 *matrix_b, f64 *matrix_u, f64 *matrix_v,
                   usize m, usize n);



void gram_schmidt_modified (f64 *Q, f64 *R, f64 *matrix_a, usize n);
void gram_schmidt_modified_2 (double *a, double *q, int n);
f64 gershgorin_test (f64 *A, usize n);
void qr_method (f64 *matrix_a, f64 *eigen_values, f64 *eigen_vectors, usize n);

#endif
