#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"

void bidiagonalisation (f64 *matrix_a, f64 *matrix_b, f64 *matrix_u, f64 *matrix_v,
                   usize m, usize n);

void gram_schmidt_modified (f64 *Q, f64 *R, f64 *matrix_a, usize n);
void gram_schmidt_modified_2 (double *a, double *q, int n);
#endif
