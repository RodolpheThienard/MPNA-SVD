#include "../include/utils.h"

u64
rdtsc ()
{
  u64 a, d;
  __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
  return (d << 32) | a;
}

void
test_matrice (f64 *M, i32 m, i32 n)
{
  srand (time (NULL));
  for (i32 i = 0; i < m; i++)
    {
      for (i32 j = 0; j < n; j++)
        M[i * n + j] = rand () % (n * m + 1);
    }
}

void
test_matrice_v1 (f64 *M, i32 m, i32 n)
{
  srand (time (NULL));
  for (i32 i = 0; i < m; i++)
    {
      M[i * n] = rand () % 11;
    }
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

f64
column_euclidean_norm (f64 *matrix, i32 rows, i32 cols, i32 col)
{
  f64 sum_of_squares = 0.0;

  for (usize i = 0; i < rows; i++)
    {
      sum_of_squares += matrix[i * cols + col] * matrix[i * cols + col];
    }

  return sqrt (sum_of_squares);
}
