#include "../include/utils.h"

u64
rdtsc ()
{
  u64 a, d;
  __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
  return (d << 32) | a;
}

void
test_matrice (f64 *M, i32 n)
{
  for (i32 i = 0; i < n * n; i++)
    M[i] = i + 1;
}

void
affiche_mat (i32 m, i32 n, f64 *mat)
{
  i32 i, j;
  for (i = 0; i < m; i++)
    {
      for (j = 0; j < n; j++)
        {
          printf ("%f ", mat[i * n + j]);
        }
      printf ("\n");
    }
}

void
transposeMatrix (f64 *matrix, i32 rows, i32 cols)
{
  f64 *tempMatrix = malloc (rows * cols * sizeof (double));

  // inversion de l'ordre
  for (i32 i = 0; i < rows; ++i)
    {
      for (i32 j = 0; j < cols; ++j)
        {
          tempMatrix[j * rows + i] = matrix[(rows - i - 1) * cols + j];
        }
    }

  for (i32 i = 0; i < rows * cols; ++i)
    {
      matrix[i] = tempMatrix[i];
    }

  free (tempMatrix);
}
