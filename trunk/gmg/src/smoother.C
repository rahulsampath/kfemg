
#include "gmg/include/smoother.h"
#include "gmg/include/gmgUtils.h"
#include <iostream>

void setupSmoother(SmootherData* data, Mat Kmat) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
  data->Kmat = Kmat;
  MatGetVecs(Kmat, PETSC_NULL, &(data->res));
  KSPCreate(comm, &(data->ksp1));
  KSPCreate(comm, &(data->ksp2));
  KSPSetType(data->ksp1, KSPCG);
  KSPSetType(data->ksp2, KSPGMRES);
  KSPSetPCSide(data->ksp1, PC_LEFT);
  KSPSetPCSide(data->ksp2, PC_RIGHT);
  PC pc1;
  PC pc2;
  KSPGetPC(data->ksp1, &pc1);
  KSPGetPC(data->ksp2, &pc2);
  PCSetType(pc1, PCSOR);
  PCSetType(pc2, PCSOR);
  KSPSetInitialGuessNonzero(data->ksp1, PETSC_TRUE);
  KSPSetInitialGuessNonzero(data->ksp2, PETSC_TRUE);
  KSPSetOperators(data->ksp1, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetOperators(data->ksp2, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetTolerances(data->ksp1, PETSC_DEFAULT, 1.0e-12, 2.0, PETSC_DEFAULT);
  KSPSetTolerances(data->ksp2, PETSC_DEFAULT, 1.0e-12, 2.0, PETSC_DEFAULT);
  KSPDefaultConvergedSetUIRNorm(data->ksp1);
  KSPDefaultConvergedSetUIRNorm(data->ksp2);
  KSPSetNormType(data->ksp1, KSP_NORM_UNPRECONDITIONED);
  KSPSetNormType(data->ksp2, KSP_NORM_UNPRECONDITIONED);
}

void destroySmoother(SmootherData* data) {
  KSPDestroy(&(data->ksp1));
  KSPDestroy(&(data->ksp2));
  VecDestroy(&(data->res));
  delete data;
}

void applySmoother(int maxIters, double tgtNorm, double currNorm, SmootherData* data, Vec in, Vec out) {
}


