#include "../include/kernels.h"
#include <cblas.h>
#include <math.h>

void
dgemm (double *C, double *A, double *B, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      {
        double r = 0;
        for (int k = 0; k < n; k++)
          r += A[i * n + k] * B[k * n + j];
        C[i * n + j] = r;
      }
}

/* Bidiagonalization of Golub-Kahan-Lanczos
   B = U * AV
   INPUT : allocated matrix a,b,u,v
   u: taille m*n
   v: taille n*n
   b : taille n
   a ; taille m*n*/
void
bidiagonalisation (f64 *matrix_a, f64 *matrix_b, f64 *matrix_u, f64 *matrix_v,
                   usize m, usize n)
{

  cblas_dscal (n, 1.0 / column_euclidean_norm (matrix_v, n, n, 0),
               &matrix_v[0],
               n); // On normalise le vecteur v1 initialisée au hasard

  for (usize i = 0; i < n; i++)
    {
      for (usize j = 0; j < m; j++)
        {
          matrix_u[j * n + i]
              = cblas_ddot (n, &matrix_a[j * n], 1, &matrix_v[i], n);

          if (i == 0)
            continue;

          // Probleme à cette étape, à la n-1 itération, avant cette opération
          // la n-1 colonne de u à des valeur différente de 0 alors que après
          // cette opération elle deviens systématiquement nul alors que
          // matrix_b[(i - 1) * n + i]et
          //  matrix_u[j * n + (i - 1)] sont non nul
          matrix_u[j * n + i]
              -= matrix_b[(i - 1) * n + i] * matrix_u[j * n + (i - 1)];
        }

      matrix_b[i * n + i] = column_euclidean_norm (matrix_u, m, n, i);

      for (usize j = 0; j < m; j++)
        matrix_u[j * n + i] /= matrix_b[i * n + i];

      if (i == n - 1)
        continue;

      for (usize j = 0; j < n; j++)
        {
          for (usize k = 0; k < m; k++)
            {
              matrix_v[j * n + i + 1]
                  += matrix_a[k * n + j] * matrix_u[k * n + i];
            }
          matrix_v[j * n + i + 1] -= matrix_b[i * n + i] * matrix_v[j * n + i];
        }
      matrix_b[i * n + (i + 1)]
          = column_euclidean_norm (matrix_v, n, n, i + 1);

      for (usize j = 0; j < n; j++)
        matrix_v[j * n + i + 1] /= matrix_b[i * n + (i + 1)];
    }
}

void
modifiedGramSchmidt (f64 *A, f64 *Q, f64 *R, usize m, usize n)
{
  for (int k = 0; k < m; ++k)
    {
      // Calcul de R
      for (int j = 0; j < k; ++j)
        {
          double dotProduct = 0.0;
          for (int i = 0; i < n; ++i)
            {
              dotProduct += Q[i * n + j] * A[i * m + k];
            }
          R[j * m + k] = dotProduct;
          for (int i = 0; i < n; ++i)
            {
              A[i * m + k] -= R[j * n + k] * Q[i * n + j];
            }
        }

      // Normalisation et mise à jour de Q
      double norm = 0.0;
      for (int i = 0; i < n; ++i)
        {
          norm += A[i * m + k] * A[i * m + k];
        }
      norm = sqrt (norm);

      R[k * m + k] = norm;

      for (int i = 0; i < n; ++i)
        {
          Q[i * n + k] = A[i * m + k] / norm;
        }
    }
}

/* Written in courses */
void
gram_schmidt_modified (f64 *a, f64 *q, f64 *r, usize n)
{
  f64 *w;
  ALLOC (w, n);
  for (usize k = 0; k < n; k++)
    {
      cblas_dcopy (n, &a[k * n], 1, w, 1);
      for (usize j = 0; j < k; j++)
        {
          r[k * n + j] = cblas_ddot (n, &q[j * n], 1, w, 1);
          f64 *rjk_qj;
          ALLOC (rjk_qj, n);
          cblas_daxpy (n, r[k * n + j], &q[j * n], 1, rjk_qj, 1);
          cblas_daxpy (n, -1, rjk_qj, 1, w, 1);
          free (rjk_qj);
        }
      r[k * n + k] = cblas_dnrm2 (n, w, 1);
      for (usize i = 0; i < n; i++)
        {
          q[k * n] = w[i] / r[k * n + k];
        }
    }
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
qr_method (f64 *matrix_a, f64 *eigen_values, f64 *eigen_vectors, usize m,
           usize n)
{
  f64 *Q, *R;
  ALLOC (Q, n * n);
  ALLOC (R, n * n);

  for (usize i = 0; i < ITERMAX; i++)
    {
      // gram_schmidt_modified (matrix_a, Q, R, n);
      modifiedGramSchmidt (matrix_a, Q, R, m, n);

      dgemm (matrix_a, R, Q, n);
      // cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, Q,
      // n,
      // R, n, 1, matrix_a, n);
      puts ("ICI");
      affiche_mat (n, n, Q);

      if (gershgorin_test (matrix_a, n) < ERR)
        break;
    }

  cblas_dcopy (n, R, n + 1, eigen_values, 1);
  cblas_dcopy (n * n, Q, 1, eigen_vectors, 1);

  free (Q);
  free (R);
}
