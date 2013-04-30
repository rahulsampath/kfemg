
#include "gmg/include/newSmoother.h"
#include "gmg/include/newRtgPC.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/fd.h"
#include <cmath>

void setupNewSmoother(NewSmootherData* data, int K, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<long long int> >& coeffs, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)(Kmat[K][currLev])), &comm);
  data->K = K;
  data->Kmat = Kmat[K][currLev];
  data->daH = da[K][currLev];
  MatGetVecs((data->Kmat), PETSC_NULL, &(data->res));
  PC pc1;
  KSPCreate(comm, &(data->ksp1));
  KSPSetType(data->ksp1, KSPCG);
  KSPSetPCSide(data->ksp1, PC_LEFT);
  KSPGetPC(data->ksp1, &pc1);
  PCSetType(pc1, PCSOR);
  KSPSetInitialGuessNonzero(data->ksp1, PETSC_TRUE);
  KSPSetOperators(data->ksp1, Kmat[K][currLev], Kmat[K][currLev], SAME_PRECONDITIONER);
  KSPSetTolerances(data->ksp1, PETSC_DEFAULT, 1.0e-12, 2.0, PETSC_DEFAULT);
  KSPDefaultConvergedSetUIRNorm(data->ksp1);
  KSPSetNormType(data->ksp1, KSP_NORM_UNPRECONDITIONED);
  PC pc2;
  KSPCreate(comm, &(data->ksp2));
  KSPSetType(data->ksp2, KSPGMRES);
  KSPSetPCSide(data->ksp2, PC_RIGHT);
  KSPGetPC(data->ksp2, &pc2);
  PCSetType(pc2, PCSOR);
  KSPSetInitialGuessNonzero(data->ksp2, PETSC_TRUE);
  KSPSetOperators(data->ksp2, Kmat[K][currLev], Kmat[K][currLev], SAME_PRECONDITIONER);
  KSPSetTolerances(data->ksp2, PETSC_DEFAULT, 1.0e-12, 2.0, PETSC_DEFAULT);
  KSPDefaultConvergedSetUIRNorm(data->ksp2);
  KSPSetNormType(data->ksp2, KSP_NORM_UNPRECONDITIONED);
  data->ksp3 = NULL;
  data->low = NULL;
  data->high = NULL;
  data->loaRhs = NULL;
  data->loaSol = NULL;
  if(K > 0) {
    data->daL = da[K - 1][currLev];
    PC pc3;
    KSPCreate(comm, &(data->ksp3));
    KSPSetType(data->ksp3, KSPGMRES);
    KSPSetPCSide(data->ksp3, PC_RIGHT);
    KSPGetPC(data->ksp3, &pc3);
    setupNewRTG(pc3, (K-1), currLev, da, coeffs, Kmat, Pmat, tmpCvec); 
    KSPSetInitialGuessNonzero(data->ksp3, PETSC_TRUE);
    KSPSetOperators(data->ksp3, Kmat[K-1][currLev], Kmat[K-1][currLev], SAME_PRECONDITIONER);
    KSPSetTolerances(data->ksp3, 0.1, 1.0e-12, 2.0, 10);
    KSPDefaultConvergedSetUIRNorm(data->ksp3);
    KSPSetNormType(data->ksp3, KSP_NORM_UNPRECONDITIONED);
    data->loa = new LOAdata;
    setupLOA(data->loa, K, (data->daL), (data->daH), coeffs);
    data->ls = new LSdata;
    setupLS(data->ls, Kmat[K][currLev]);
    VecDuplicate((data->res), &(data->low));
    VecDuplicate((data->res), &(data->high));
    MatGetVecs((Kmat[K-1][currLev]), &(data->loaSol), &(data->loaRhs));
  }
}

void destroyNewSmoother(NewSmootherData* data) {
  KSPDestroy(&(data->ksp1));
  KSPDestroy(&(data->ksp2));
  if((data->K) > 0) {
    KSPDestroy(&(data->ksp3));
    destroyLOA(data->loa);
    destroyLS(data->ls);
    VecDestroy(&(data->low));
    VecDestroy(&(data->high));
    VecDestroy(&(data->loaRhs));
    VecDestroy(&(data->loaSol));
  }
  VecDestroy(&(data->res));
  delete data;
}

void applyNewSmoother(int maxIters, double tgtNorm, double currNorm,
    NewSmootherData* data, Vec in, Vec out) {
  for(int iter = 0; iter < maxIters; ++iter) {
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= tgtNorm) {
      break;
    }
    KSPSetTolerances(data->ksp1, (tgtNorm/currNorm), PETSC_DEFAULT, PETSC_DEFAULT, (iter + 1));
    KSPSolve(data->ksp1, in, out);
    computeResidual(data->Kmat, out, in, data->res);
    VecNorm(data->res, NORM_2, &currNorm);
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= tgtNorm) {
      break;
    }
    KSPSetTolerances(data->ksp2, (tgtNorm/currNorm), PETSC_DEFAULT, PETSC_DEFAULT, (iter + 1));
    KSPSolve(data->ksp2, in, out);
    computeResidual(data->Kmat, out, in, data->res);
    VecNorm(data->res, NORM_2, &currNorm);
    if((data->K) > 0) {
      PetscInt dim;
      DMDAGetInfo(data->daH, &dim, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
          PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
      PetscInt xs, ys, zs;
      PetscInt nx, ny, nz;
      DMDAGetCorners(data->daH, &xs, &ys, &zs, &nx, &ny, &nz);
      for(int it = 0; it < (iter + 1); ++it) {
        if(currNorm <= 1.0e-12) {
          break;
        }
        if(currNorm <= tgtNorm) {
          break;
        }
        applyLOA(data->loa, data->res, data->loaRhs);
        VecZeroEntries(data->loaSol);
        KSPSolve(data->ksp3, data->loaRhs, data->loaSol);
        VecZeroEntries(data->low);
        if(dim == 1) {
          PetscScalar** inArr;
          PetscScalar** outArr;
          DMDAVecGetArrayDOF(data->daL, data->loaSol, &inArr);
          DMDAVecGetArrayDOF(data->daH, data->low, &outArr);
          for(int xi = xs; xi < (xs + nx); ++xi) {
            for(int d = 0; d < (data->K); ++d) {
              outArr[xi][d] = inArr[xi][d];
            }//end d
          }//end xi 
          DMDAVecRestoreArrayDOF(data->daL, data->loaSol, &inArr);
          DMDAVecRestoreArrayDOF(data->daH, data->low, &outArr);
        } else if(dim == 2) {
          PetscScalar*** inArr;
          PetscScalar*** outArr;
          DMDAVecGetArrayDOF(data->daL, data->loaSol, &inArr);
          DMDAVecGetArrayDOF(data->daH, data->low, &outArr);
          for(int yi = ys; yi < (ys + ny); ++yi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dy = 0, d = 0; dy < (data->K); ++dy) {
                for(int dx = 0; dx < (data->K); ++dx, ++d) {
                  outArr[yi][xi][(dy*(data->K + 1)) + dx] = inArr[yi][xi][d];
                }//end dx
              }//end dy
            }//end xi 
          }//end yi 
          DMDAVecRestoreArrayDOF(data->daL, data->loaSol, &inArr);
          DMDAVecRestoreArrayDOF(data->daH, data->low, &outArr);
        } else {
          PetscScalar**** inArr;
          PetscScalar**** outArr;
          DMDAVecGetArrayDOF(data->daL, data->loaSol, &inArr);
          DMDAVecGetArrayDOF(data->daH, data->low, &outArr);
          for(int zi = zs; zi < (zs + nz); ++zi) {
            for(int yi = ys; yi < (ys + ny); ++yi) {
              for(int xi = xs; xi < (xs + nx); ++xi) {
                for(int dz = 0, d = 0; dz < (data->K); ++dz) {
                  for(int dy = 0; dy < (data->K); ++dy) {
                    for(int dx = 0; dx < (data->K); ++dx, ++d) {
                      outArr[zi][yi][xi][(((dz*(data->K + 1)) + dy)*(data->K + 1)) + dx] = inArr[zi][yi][xi][d];
                    }//end dx
                  }//end dy
                }//end dz
              }//end xi 
            }//end yi 
          }//end zi 
          DMDAVecRestoreArrayDOF(data->daL, data->loaSol, &inArr);
          DMDAVecRestoreArrayDOF(data->daH, data->low, &outArr);
        }
        VecZeroEntries(data->high);
        applyFD((data->daH), (data->K), (data->low), (data->high));
        double a[2];
        double normUpdate = applyLS(data->ls, data->res, data->low, data->high, a); 
        VecAXPBYPCZ(out, a[0], a[1], 1.0, data->low, data->high);
        currNorm = std::sqrt((currNorm*currNorm) + normUpdate);
      }//end it
    }
  }//end iter
}


