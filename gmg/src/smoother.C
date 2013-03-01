
#include "gmg/include/smoother.h"

void setupSmootherData(SmootherData* data, Mat Kmat) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
  data->maxIts = 4;
  PetscOptionsGetInt(PETSC_NULL, "-smootherMaxIts", &(data->maxIts), PETSC_NULL);
  data->tol = 0.5;
  PetscOptionsGetReal(PETSC_NULL, "-smootherTol", &(data->tol), PETSC_NULL);
}

void destroySmootherData(SmootherData* data) {
  KSPDestroy(&(data->ksp1));
  KSPDestroy(&(data->ksp2));
  VecDestroy(&(data->res));
  delete data;
}

void applySmoother(SmootherData* data, Vec in, Vec out) {
}


