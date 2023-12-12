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
