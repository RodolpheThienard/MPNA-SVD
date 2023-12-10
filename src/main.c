#include "../include/utils.h"
#include <cblas-openblas.h>
#include <cblas.h>

i32
main (i32 argc, char *argv[])
{
#ifdef DEBUG
  printf ("DEBUG MODE ON\n");
#endif

  // definition
  i32 size_m = atoi (argv[1]);
  i32 size_n = atoi (argv[2]);
  f64 *matrix_m, *matrix_u, *matrix_v, *matrix_vt, *eval_v, *eval, *evec,
      *sigma, *res, *res2, *matrix_ut;

  // matrix allocation
  ALLOC (sizeof (f64) * size_m * size_n, matrix_m);
  ALLOC (sizeof (f64) * size_m * size_m, matrix_u);
  ALLOC (sizeof (f64) * size_m * size_m, matrix_ut);
  ALLOC (sizeof (f64) * size_n * size_n, matrix_v);
  ALLOC (sizeof (f64) * size_n * size_n, matrix_vt);
  ALLOC (sizeof (f64) * size_m, eval);
  ALLOC (sizeof (f64) * size_n, eval_v);
  ALLOC (sizeof (f64) * size_m * size_m, evec);
  ALLOC (sizeof (f64) * size_m * size_n, sigma);
  ALLOC (sizeof (f64) * size_m * size_n, res);
  ALLOC (sizeof (f64) * size_m * size_n, res2);

  test_matrice (matrix_m, size_m);

#ifdef DEBUG
  printf ("Matrice M\n");
  affiche_mat (size_m, size_n, matrix_m);
#endif
  // compute U
  cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, size_n, size_n, size_m,
               1.0, matrix_m, size_m, matrix_m, size_m, 0.0, matrix_u, size_n);
  // Compute V
  cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, size_n, size_n, size_m,
               1.0, matrix_m, size_m, matrix_m, size_m, 0.0, matrix_v, size_n);

  // Compute eigen vectors / values
  i32 info = LAPACKE_dsyev (LAPACK_COL_MAJOR, 'V', 'U', size_m, matrix_u,
                            size_m, eval);
  transposeMatrix (matrix_u, size_m, size_n);

  // Vérification des erreurs
  if (info > 0)
    {
      fprintf (stderr, "Erreur dans le calcul des valeurs propres\n");
      return 1;
    }

#ifdef DEBUG
  printf ("Matrice U\n");
  affiche_mat (size_m, size_m, matrix_u);
#endif

  // Beginning of measurement
#ifdef BENCHMARK
  u64 t0 = rdtsc ();
#endif

  for (int i = 0; i < size_m; i++)
    {
      f64 eval_i = eval[i];
      if (eval_i > 0.0)
        {
          sigma[(size_n * size_m) - (i * size_n + i + 1)] = sqrt (eval[i]);
        }
      else
        {
          sigma[i * size_n + i] = 0.0;
        }
    }

#ifdef DEBUG
  printf ("Matrice sigma\n");
  affiche_mat (size_m, size_n, sigma);
#endif
  cblas_dgemm (CblasRowMajor, CblasTrans, CblasTrans, size_m, size_n, size_n,
               1.0, evec, size_m, sigma, size_n, 0.0, res, size_m);

  i32 info_v = LAPACKE_dsyev (LAPACK_COL_MAJOR, 'V', 'U', size_n, matrix_v,
                              size_n, eval);
  // Vérification des erreurs
  if (info_v > 0)
    {
      fprintf (stderr, "Erreur dans le calcul des valeurs propres\n");
      return 1;
    }

  // transpose evec matrix_vt
  transposeMatrix (matrix_v, size_m, size_n);

#ifdef DEBUG
  printf ("Matrice V*\n");
  affiche_mat (size_n, size_n, matrix_v);
#endif

// End of measurement + print
#ifdef BENCHMARK
  u64 t1 = rdtsc ();
  printf ("total cycles : %ld\n", t1 - t0);
#endif

  return 0;
}
