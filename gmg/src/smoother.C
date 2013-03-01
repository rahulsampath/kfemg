
#include "gmg/include/smoother.h"
#include "gmg/include/gmgUtils.h"

void setupSmootherData(SmootherData* data, Mat Kmat) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
  data->maxIts = 4;
  PetscOptionsGetInt(PETSC_NULL, "-smootherMaxIts", &(data->maxIts), PETSC_NULL);
  data->tol = 0.5;
  PetscOptionsGetReal(PETSC_NULL, "-smootherTol", &(data->tol), PETSC_NULL);
  data->Kmat = Kmat;
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
  PCSetType(pc1, PCNONE);
  PCSetType(pc2, PCNONE);
  KSPSetInitialGuessNonzero(data->ksp1, PETSC_TRUE);
  KSPSetInitialGuessNonzero(data->ksp2, PETSC_TRUE);
  KSPSetOperators(data->ksp1, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetOperators(data->ksp2, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetTolerances(data->ksp1, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
  KSPSetTolerances(data->ksp2, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
  KSPSetOptionsPrefix(data->ksp1, "smooth1_");
  KSPSetOptionsPrefix(data->ksp2, "smooth2_");
  KSPSetFromOptions(data->ksp1);
  KSPSetFromOptions(data->ksp2);
  MatGetVecs(Kmat, PETSC_NULL, &(data->res));
}

void destroySmootherData(SmootherData* data) {
  KSPDestroy(&(data->ksp1));
  KSPDestroy(&(data->ksp2));
  VecDestroy(&(data->res));
  delete data;
}

void applySmoother(SmootherData* data, Vec in, Vec out) {
}


