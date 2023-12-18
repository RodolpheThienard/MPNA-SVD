#include "../include/kernels.h"
#include "../include/utils.h"
#include <bits/time.h>
#include <cblas-openblas.h>
#include <cblas.h>
#include <time.h>

i32
main (i32 argc, char *argv[])
{
#ifdef DEBUG
  printf ("DEBUG MODE ON\n");
#endif

  // definition
  i32 size_m = atoi (argv[1]);
  i32 size_n = atoi (argv[2]);
  i32 size_min = MIN (size_n, size_m);
  f64 *matrix_a, *matrix_u1, *matrix_u2, *matrix_c, *matrix_b1, *matrix_b2,
      *matrix_v1, *matrix_v2, *eval_v, *eval, *sigma, *res, *res2, *matrix_ut;

  // matrix allocation
  ALLOC (matrix_a, size_m * size_n);
  ALLOC (matrix_u1, size_m * size_n);
  ALLOC (matrix_u2, size_m * size_n);
  ALLOC (matrix_b1, size_n * size_n);
  ALLOC (matrix_b2, size_n * size_n);
  ALLOC (matrix_v1, size_n * size_n);
  ALLOC (matrix_v2, size_n * size_n);
  ALLOC (matrix_c, size_n * size_n);
  ALLOC (eval, size_min); // Il y a min(n,m) valeur propre
  ALLOC (eval_v, size_n);
  ALLOC (sigma, size_m * size_n);
  ALLOC (res, size_m * size_n);
  ALLOC (res2, size_m * size_n);

  // Initialisation matrice A
  test_matrice (matrix_a, size_m, size_n);

  // Initialisation vecteur v1
  test_matrice_v1 (matrix_v1, size_n, size_n);

#ifdef DEBUG
  printf ("Matrice A\n");
  affiche_mat (size_m, size_n, matrix_a);
#endif

  // Beginning of measurement
#ifdef BENCHMARK
  struct timespec t0, t1;
  clock_gettime (CLOCK_MONOTONIC_RAW, &t0);
  u64 r0 = rdtsc ();
#endif

  // Bidiagonalisation de C = U1*AV1
  bidiagonalisation (matrix_a, matrix_c, matrix_u1, matrix_v1, size_m, size_n);

#ifdef DEBUG
  printf ("Matrice U1\n");
  affiche_mat (size_m, size_n, matrix_u1);
  printf ("Matrice V1\n");
  affiche_mat (size_n, size_n, matrix_v1);
  printf ("Matrice C\n");
  affiche_mat (size_n, size_n, matrix_c);
#endif

  // Cacul du SVD de C = U2 SIGMA V2*
  // Compute U2 = vecteur propre de B1=CCt
  cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, size_n, size_n, size_n,
               1.0, matrix_c, size_n, matrix_c, size_n, 0.0, matrix_b1,
               size_n);
  qr_method (matrix_b1, eval, matrix_u2, size_n, size_n);

  // Compute V2 = vecteur propre de B2=CtC
  cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, size_n, size_n, size_n,
               1.0, matrix_c, size_n, matrix_c, size_n, 0.0, matrix_b2,
               size_n);
  qr_method (matrix_b2, eval_v, matrix_v2, size_n, size_n);

  // Sigma final
  for (int i = 0; i < size_min; i++)
    {
      sigma[(i * size_n + i)] = sqrt (eval[i]);
    }

#ifdef DEBUG
  printf ("Matrice U2\n");
  affiche_mat (size_n, size_n, matrix_u2);
  printf ("Matrice V2\n");
  affiche_mat (size_n, size_n, matrix_v2);
#endif

#ifdef DEBUG
  printf ("Matrice sigma\n");
  affiche_mat (size_m, size_n, sigma);
#endif

  // End of measurement + print
#ifdef BENCHMARK
  u64 r1 = rdtsc ();
  clock_gettime (CLOCK_MONOTONIC_RAW, &t1);
  printf ("total cycles : %ld\n", r1 - r0);
  printf ("total time elapsed : %lf\n",
          (t1.tv_sec + t1.tv_nsec * 1e-9) - (t0.tv_sec + t0.tv_nsec * 1e-9));
#endif

  return 0;
}
