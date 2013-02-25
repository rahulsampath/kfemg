
#include "gmg/include/lsFitType3PC.h"
#include "gmg/include/gmgUtils.h"
#include <iostream>

void setupLSfitType3PC(PC pc, Mat Kmat, Mat reducedMat, int K, int Nx,
    std::vector<long long int>& coeffsCK, std::vector<long long int>& coeffsC0) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
  LSfitType3Data* data = new LSfitType3Data;
  data->K = K;
  data->Nx = Nx;
  data->coeffsCK = &coeffsCK;
  data->coeffsC0 = &coeffsC0; 
  data->Kmat = Kmat;
  MatGetVecs(Kmat, &(data->err), &(data->res));
  VecDuplicate((data->res), &(data->tmp1));
  VecDuplicate((data->res), &(data->tmp2));
  VecDuplicate((data->res), &(data->fTilde));
  MatGetVecs(reducedMat, &(data->reducedSol), &(data->reducedRhs));
  KSPCreate(comm, &(data->reducedSolver));
  KSPSetType((data->reducedSolver), KSPCG);
  KSPSetPCSide((data->reducedSolver), PC_LEFT);
  PC tmpPC;
  KSPGetPC((data->reducedSolver), &tmpPC);
  PCSetType(tmpPC, PCNONE);
  KSPSetOperators((data->reducedSolver), reducedMat, reducedMat, SAME_PRECONDITIONER);
  KSPSetInitialGuessNonzero((data->reducedSolver), PETSC_FALSE);
  KSPSetTolerances((data->reducedSolver), 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
  KSPSetOptionsPrefix((data->reducedSolver), "C0_");
  KSPSetFromOptions(data->reducedSolver);
  PCShellSetContext(pc, data);
  PCShellSetName(pc, "MyLSPC");
  PCShellSetApply(pc, &applyLSfitType3PC);
  PCShellSetDestroy(pc, &destroyLSfitType3PC);
} 

PetscErrorCode destroyLSfitType3PC(PC pc) {
  LSfitType3Data* data;
  PCShellGetContext(pc, (void**)(&data));
  KSPDestroy(&(data->reducedSolver));
  VecDestroy(&(data->res));
  VecDestroy(&(data->err));
  VecDestroy(&(data->fTilde));
  VecDestroy(&(data->tmp1));
  VecDestroy(&(data->tmp2));
  VecDestroy(&(data->reducedRhs));
  VecDestroy(&(data->reducedSol));
  delete data;
  return 0;
}



