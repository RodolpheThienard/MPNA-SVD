#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef double f64;
typedef int32_t i32;
typedef uint64_t u64;
typedef size_t usize;

#define ALLOC(X, n)                                                           \
  do                                                                          \
    {                                                                         \
      X = malloc (sizeof (*X) * n);                                           \
      if (!X)                                                                 \
        {                                                                     \
          perror ("malloc");                                                  \
          exit (1);                                                           \
        }                                                                     \
      memset (X, 0, sizeof (*X) * n);                                         \
    }                                                                         \
  while (0)

void transposeMatrix (f64 *matrix, i32 rows, i32 cols);
void affiche_mat (i32 m, i32 n, f64 *mat);
u64 rdtsc ();
void test_matrice (f64 *, i32, i32);
void test_matrice_v1 (f64 *M, i32 m, i32 n);
