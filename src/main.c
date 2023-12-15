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
  f64 *matrix_a, *matrix_u1,*matrix_u2,*matrix_c,*matrix_b1,*matrix_b2, *matrix_v1,*matrix_v2,  *eval_v, *eval, 
      *sigma, *res, *res2, *matrix_ut;

  // matrix allocation
  ALLOC (sizeof (f64) * size_m * size_n, matrix_a);
  ALLOC (sizeof (f64) * size_m * size_n, matrix_u1);
  ALLOC (sizeof (f64) * size_m * size_n, matrix_u2);
  ALLOC (sizeof (f64) * size_n * size_n, matrix_c);
  ALLOC (sizeof (f64) * size_n * size_n, matrix_b1);
  ALLOC (sizeof (f64) * size_n * size_n, matrix_b2);
  ALLOC (sizeof (f64) * size_n * size_n, matrix_v1);
  ALLOC (sizeof (f64) * size_n * size_n, matrix_v2);
  ALLOC (sizeof (f64) * size_n, eval);
  ALLOC (sizeof (f64) * size_n, eval_v);
  ALLOC (sizeof (f64) * size_m * size_n, sigma);
  ALLOC (sizeof (f64) * size_m * size_n, res);
  ALLOC (sizeof (f64) * size_m * size_n, res2);


  test_matrice (matrix_a, size_m, size_n);



  // Initialisation vecteur v1
  test_matrice_v1(matrix_v1, size_n, size_n);

  #ifdef DEBUG
  printf ("Matrice A\n");
  affiche_mat (size_m, size_n, matrix_a);
  #endif

  // Bidiagonalisation de C = U1*AV1
  bidiagonalisation (matrix_a, matrix_c, matrix_u1, matrix_v1,
                   size_m, size_n);

  #ifdef DEBUG
  printf ("Matrice U1\n");
  affiche_mat (size_m, size_n, matrix_u1);
  printf ("Matrice V1\n");
  affiche_mat (size_n, size_n, matrix_v1);
  printf ("Matrice C\n");
  affiche_mat (size_n, size_n, matrix_c);
  #endif


 
  //Cacul du SVD de C = U2 SIGMA V2*

  // Compute U2 = vecteur propre de B1=CCt
  cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, size_n, size_n, size_n,
               1.0, matrix_c, size_n, matrix_c, size_n, 0.0, matrix_b1, size_n);
  qr_method (matrix_b1, eval, matrix_u2, size_n);


  // Compute V2 = vecteur propre de B2=CtC
  cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, size_n, size_n, size_n,
               1.0, matrix_c, size_n, matrix_c, size_n, 0.0, matrix_b2, size_n);
  qr_method (matrix_b2, eval_v, matrix_v2, size_n);

  //Sigma final 
  for (int i = 0; i < size_n; i++)
    {
      f64 eval_i = eval[i];
      if (eval_i > 0.0)
        {
          sigma[(size_n * size_m) - (i * size_m + i + 1)] = sqrt (eval[i]);
        }
      else
        {
          sigma[i * size_m + i] = 0.0;
        }
    }
    transposeMatrix (sigma, size_n, size_m);
    #ifdef DEBUG
   // printf ("Matrice Sigma\n");
    //affiche_mat (size_m, size_n, sigma);
        printf ("Matrice U2\n");
    affiche_mat (size_n, size_n, matrix_u2);
    printf ("Matrice V2\n");
    affiche_mat (size_n, size_n, matrix_v2);
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
  /*cblas_dgemm (CblasRowMajor, CblasTrans, CblasTrans, size_m, size_n, size_n,
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
*/
// End of measurement + print
#ifdef BENCHMARK
  u64 t1 = rdtsc ();
  printf ("total cycles : %ld\n", t1 - t0);
#endif

  return 0;
}
