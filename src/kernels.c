#include "../include/kernels.h"
#include <cblas.h>
#include <math.h>

/* Bidiagonalization of Golub-Kahan-Lanczos
   B = U * AV
   INPUT : allocated matrix a,b,u,v */
void
bidiagonalisation (f64 *matrix_a, f64 *matrix_b, f64 *matrix_u, f64 *matrix_v,
                   usize m, usize n)
{
  for (usize i = 0; i < m; i++)
    {
      for (usize j = 0; j < m; j++)
        {
          matrix_u[j * m + i]
              = cblas_ddot (n, &matrix_a[j * n], 1, &matrix_v[i], n);
          if (i == 0)
            continue;
          matrix_u[j * m + i]
              -= matrix_b[(i - 1) * n + i] * matrix_u[j * m + i - 1];
        }

      matrix_b[i * n + i] = cblas_dnrm2 (m, &matrix_u[i], m);

      for (usize j = 0; j < m; j++)
        matrix_u[j * m + i] /= matrix_b[i * n + i];

      if (i == n - 1)
        continue;

      for (usize j = 0; j < m; j++)
        matrix_v[j * n + i + 1]
            = cblas_ddot (m, &matrix_a[j], n, &matrix_u[i], m)
              - matrix_b[i * n + i] * matrix_v[j * n + i];

      matrix_b[i * n + i + 1] = cblas_dnrm2 (n, &matrix_v[i + 1], n);

      for (usize j = 0; j < m; j++)
        matrix_v[j * n + i + 1] /= matrix_b[i * n + i + 1];
    }
}

/* TODO
 */
void
gram_schmidt_modified (f64 *Q, f64 *R, f64 *matrix_a, usize n)
{
  for (usize i = 0; i < n; i++)
    {
      for (usize j = 0; j < n; j++)
        Q[j * n + i] = matrix_a[j * n + i];

      for (usize j = 0; j < n; j++)
        {
          R[j * n + i] = cblas_ddot (n, Q + j * n, 1, Q + i * n, 1);

          for (usize k = 0; k < n; k++)
            Q[k * n + i] -= R[j * n + i] * Q[k * n + j];
        }
      R[i * n + i] = cblas_dnrm2 (n, Q + i * n, 1);

      for (usize j = 0; j < n; j++)
        Q[j * n + i] /= R[i * n + i];
    }
}

/* Written in courses */
void
gram_schmidt_modified_2 (double *a, double *q, int n)
{
  double *r = calloc (n * n, sizeof (double));
  double *w = calloc (n, sizeof (double));
  for (int k = 0; k < n; k++)
    {
      cblas_dcopy (n, &a[k * n], 1, w, 1);
      for (int j = 0; j < k; j++)
        {
          r[k * n + j] = cblas_ddot (n, &q[j * n], 1, w, 1);
          double *rjk_qj = calloc (n, sizeof (double));
          cblas_daxpy (n, r[k * n + j], &q[j * n], 1, rjk_qj, 1);
          cblas_daxpy (n, -1, rjk_qj, 1, w, 1);
          free (rjk_qj);
        }
      r[k * n + k] = cblas_dnrm2 (n, w, 1);
      for (int i = 0; i < n; i++)
        {
          q[k * n] = w[i] / r[k * n + k];
        }
    }
  free (r);
  free (w);
}

/* TODO
 */
f64
gershgorin_test (f64 *A, usize n)
{
  f64 err_max = 0;
  for (usize i = 0; i < n; i++)
    {
      f64 err = 0;
      for (usize j = i + 1; j < n; j++)
        err += fabs (A[j * n + i]);
      if (err > err_max)
        err_max = err;
    }
  return err_max;
}

/* TODO
 */
void
qr_method (f64 *matrix_a, f64 *eigen_values, f64 *eigen_vectors, usize n)
{
  f64 *Q, *R;

  ALLOC (sizeof (f64) * n * n, Q);
  ALLOC (sizeof (f64) * n * n, R);

  for (usize i = 0; i < ITERMAX; i++)
    {
      gram_schmidt_modified (Q, R, matrix_a, n);

      cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Q, n,
                   R, n, 1, matrix_a, n);

      if (gershgorin_test (matrix_a, n) < ERR)
        break;
    }
  cblas_dcopy (n, R, n + 1, eigen_values, 1);
  cblas_dcopy (n, Q, n, eigen_vectors, n);

  free (Q);
  free (R);
}
