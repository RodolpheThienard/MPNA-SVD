#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"
#define ITERMAX 300
#define ERR 0.001

void dgemm (double *C, double *A, double *B, int n);
void bidiagonalisation (f64 *matrix_a, f64 *matrix_b, f64 *matrix_u,
                        f64 *matrix_v, usize m, usize n);

void qr_method (f64 *matrix_a, f64 *eigen_values, f64 *eigen_vectors, usize m,
                usize n);

#endif
