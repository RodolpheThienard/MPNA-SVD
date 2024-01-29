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
  i32 size_max = MAX(size_m, size_n);
  lapack_int info;
  f64 *matrix_a, *matrix_tmp, *matrix_a_copy, *matrix_a_ref, *matrix_a_compute,*matrix_u_compute,  *matrix_u_ref, *matrix_atr,  *matrix_u1, *matrix_u2, *matrix_c, *matrix_b1, *matrix_b2,
      *matrix_v1, *matrix_v2, *matrix_v_compute, *matrix_v_ref, *eval_v, *eval, *sigma, *res, *res2, *matrix_ut, *eval_ref, *superb, *sigma_ref, *matrix_tmp_ref, error_lapack, error_compute, error_vs;

  // matrix allocation
  ALLOC (matrix_a, size_m * size_n);
  ALLOC (matrix_a_copy, size_m * size_n);
  ALLOC (matrix_a_compute, size_m * size_n);
  ALLOC (matrix_a_ref, size_m * size_n);
  ALLOC (matrix_u1, size_min * size_max);
  ALLOC (matrix_u_compute, size_min * size_max);
  ALLOC (matrix_u_ref, size_m * size_min);
  ALLOC (matrix_tmp, size_min * size_max);
  ALLOC (matrix_tmp_ref, size_min * size_max);
  ALLOC (matrix_u2, size_max * size_max);
  ALLOC (matrix_b1, size_max * size_max);
  ALLOC (matrix_b2, size_max * size_max);
  ALLOC (matrix_v1, size_max * size_max);
  ALLOC (matrix_v_compute, size_max * size_max);
  ALLOC (matrix_v_ref, size_n * size_n);
  ALLOC (matrix_v2, size_max * size_max);
  ALLOC (matrix_c, size_max * size_max);
  ALLOC (eval, size_min);   //Il y a min(n,m) valeur propre
  ALLOC (eval_v, size_max);
  ALLOC (superb, (size_min-1));
  ALLOC (sigma, size_max * size_max);
  ALLOC (sigma_ref, size_min * size_n);
  ALLOC (eval_ref, size_min);
  ALLOC (res, size_m * size_n);
  ALLOC (res2, size_m * size_n);

  // Initialisation matrice A
  Matrix_initialization (matrix_a, size_m, size_n);

  // Initialisation vecteur v1
  Initialization_v1 (matrix_v1, size_n, size_n);

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

  //Si m<=n alors on prend A sinon on prend A^t
  if(size_m>size_n)
  {
    ALLOC (matrix_atr, size_n * size_m);
    transposeMatrix (matrix_a, matrix_atr,size_m, size_n);
    bidiagonalisation (matrix_atr, matrix_c, matrix_u1, matrix_v1, size_min, size_max);
    #ifdef DEBUG
      printf ("Matrice Atransposée\n");
      affiche_mat (size_n, size_m, matrix_atr);
    #endif
  }

  else
  {
    bidiagonalisation (matrix_a, matrix_c, matrix_u1, matrix_v1, size_min, size_max);
  }

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
  cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, size_max, size_max, size_max,
               1.0, matrix_c, size_max, matrix_c, size_max, 0.0, matrix_b1,
               size_max);
  qr_method (matrix_b1, eval, matrix_u2, size_max, size_max);

  // Compute V2 = vecteur propre de B2=CtC
  cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, size_max, size_max, size_max,
               1.0, matrix_c, size_max, matrix_c, size_max, 0.0, matrix_b2,
               size_max);
  qr_method (matrix_b2, eval_v, matrix_v2, size_max, size_max);


  // Sigma final
  for (int i = 0; i < size_max; i++)
    {
          sigma[(i * size_max + i)] = sqrt (eval[i]);
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

//Calcul erreur relative avec comme réference svd lapack

//Première étape calculer la matrice de résultat : U1U2sigmaV2*V1* qui doit etre normalement égale A 
// Calcul U=U1U2
cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, size_min, size_max, size_max,
               1.0, matrix_u1, size_max, matrix_u2, size_max, 0.0, matrix_u_compute,
               size_max);

//Calcul V=V2^tV1^t
cblas_dgemm (CblasRowMajor, CblasTrans, CblasTrans, size_max, size_max, size_max,
               1.0, matrix_v2, size_max, matrix_v1, size_max, 0.0, matrix_v_compute,
               size_max);

//Calcul tmp=Usigma
cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, size_min, size_max, size_max,
               1.0, matrix_u_compute, size_max, sigma, size_max, 0.0, matrix_tmp,
               size_max);

//Calcul A_compute=tmpV
cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, size_min, size_max, size_max,
               1.0, matrix_tmp, size_max, matrix_v_compute, size_max, 0.0, matrix_a_compute,
               size_max);

cblas_daxpby(size_m * size_n, 1.0, matrix_a, 1,-1.0, matrix_a_compute, 1);


//Deuxième étape calculer le SVD de A grace à lapack
//On utilise une copie de la matrice A car la fonction de lapacke modifie la matrice A

copy_matrix(size_m,size_n,matrix_a_copy,matrix_a);
info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', size_m, size_n, matrix_a_copy, size_n, eval_ref, matrix_u_ref, size_m, matrix_v_ref, size_n,superb);
  if (info > 0)
  {
    fprintf(stderr, "Erreur lors du calcul de la SVD.\n");
    return 1;
  }

// Sigma final
  for (int i = 0; i < size_min; i++)
    {
          sigma_ref[(i * size_n + i)] = eval_ref[i];
    }


//Calcul de U_ref*Sigma_ref*Vt_ref
//Calcul tmp=U_refsigma_ref
cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, size_m, size_n, size_min,
               1.0, matrix_u_ref, size_min, sigma_ref, size_n, 0.0, matrix_tmp_ref,
               size_n);

//Calcul A_ref=tmp_refVt_ref
cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, size_m, size_n, size_n,
               1.0, matrix_tmp_ref, size_n, matrix_v_ref, size_n, 0.0, matrix_a_ref,
               size_n);
cblas_daxpby(size_m * size_n, 1.0, matrix_a, 1,-1.0, matrix_a_ref, 1);


//Dernière étape calcul de l'erreur relative 
error_lapack = ((Norm_Frobenius(size_m,size_n,matrix_a_ref)) / (Norm_Frobenius(size_m, size_n, matrix_a)))*100;
error_compute = ((Norm_Frobenius(size_m,size_n,matrix_a_compute)) / (Norm_Frobenius(size_m, size_n, matrix_a)))*100;
//Erreur du calcul des valeur singulière
for (i32 i = 0; i < size_min; i++) 
{
  error_vs = fabs(eval[i] - eval_ref[i]) / fabs(eval_ref[i]);
}


#ifdef DEBUG
  printf ("Matrice A_copy\n");
  affiche_mat (size_m, size_n, matrix_a_copy);
  printf ("Matrice A_ref\n");
  affiche_mat (size_m, size_n, matrix_a_ref);
  printf("Erreur relative avec la svd lapack: %f\n", error_lapack);
  printf("Erreur relative de notre svd: %f\n", error_compute);
  printf("Erreur relative valeur singulière comparé à celle trouver par lapacke: %f\n", error_vs);

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
