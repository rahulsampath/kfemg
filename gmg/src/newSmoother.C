
#include "gmg/include/newSmoother.h"
#include "gmg/include/gmgUtils.h"

void setupNewSmoother(NewSmootherData* data, int K, int currLev,
    std::vector<std::vector<DM> >& da, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)(Kmat[K][currLev])), &comm);
  data->K = K;
  data->Kmat = Kmat[K][currLev];
  data->da = da[K][currLev];
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
  if(K > 0) {
    PC pc3;
    KSPCreate(comm, &(data->ksp3));
    KSPSetType(data->ksp3, KSPGMRES);
    KSPSetPCSide(data->ksp3, PC_RIGHT);
    KSPGetPC(data->ksp3, &pc3);
    setupNewRTG(pc3, (K-1), currLev, da, Kmat, Pmat, tmpCvec); 
    KSPSetInitialGuessNonzero(data->ksp3, PETSC_TRUE);
    KSPSetOperators(data->ksp3, Kmat[K-1][currLev], Kmat[K-1][currLev], SAME_PRECONDITIONER);
    KSPSetTolerances(data->ksp3, PETSC_DEFAULT, 1.0e-12, 2.0, PETSC_DEFAULT);
    KSPDefaultConvergedSetUIRNorm(data->ksp3);
    KSPSetNormType(data->ksp3, KSP_NORM_UNPRECONDITIONED);
    data->ls = new LSdata;
    setupLS(data->ls);
    data->loa = new LOAdata;
    setupLOA(data->loa);
  }
}

void destroyNewSmoother(NewSmootherData* data) {
  KSPDestroy(&(data->ksp1));
  KSPDestroy(&(data->ksp2));
  if(data->ksp3 != NULL) {
    KSPDestroy(&(data->ksp3));
    destroyLS(data->ls);
    destroyLOA(data->loa);
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
  }//end iter
}


