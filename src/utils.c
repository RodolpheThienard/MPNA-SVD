#include "../include/utils.h"

u64
rdtsc ()
{
  u64 a, d;
  __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
  return (d << 32) | a;
}

void
Matrix_initialization (f64 *M, i32 m, i32 n)
{
  srand (time (NULL));
  for (i32 i = 0; i < m; i++)
    {
      for (i32 j = 0; j < n; j++)
        M[i * n + j] = rand () % (n * m + 1);
    }
}

void
Initialization_v1 (f64 *M, i32 m, i32 n)
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

void
transposeMatrix (f64 *matrix,f64* matrix_tr, i32 rows, i32 cols)
{

  // inversion de l'ordre
  for (i32 i = 0; i < rows; ++i)
    {
      for (i32 j = 0; j < cols; ++j)
        {
          matrix_tr[j * rows + i] = matrix[i* cols + j];
        }
    }
}


f64 Norm_Frobenius(i32 rows, i32 cols, f64 *matrix)
{
    f64 result=0.0;

    for(i32 i=0; i<rows; i++)
    {
        for(i32 j=0; j<cols;j++)
            result += matrix[i*cols+j]*matrix[i*cols+j];
        
    }
    return sqrt(result);
}

void copy_matrix(i32 rows, i32 cols, f64* matrix_copy, f64* matrix) {
    for (i32 i = 0; i < rows*cols; i++) {
        matrix_copy[i] = matrix[i];
    }
}