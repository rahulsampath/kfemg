
#include "gmg/include/rtgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/intergrid.h"
#include <iostream>

#ifdef DEBUG
#include <cassert>
#endif

void setupRTG(PC pc, int K, int currLev, std::vector<DM>& da, std::vector<Mat>& Kmat,
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec) {
#ifdef DEBUG
  assert(currLev > 0);
#endif
  RTGdata* data = new RTGdata; 
  data->K = K;
  data->da = da[currLev];
  data->Kmat = Kmat[currLev];
  data->Pmat = Pmat[currLev - 1];
  data->tmpCvec = tmpCvec[currLev - 1];
  data->sData = new SmootherData;
  setupSmoother(data->sData, data->Kmat);
  MatGetVecs(Kmat[currLev], PETSC_NULL, &(data->res));
  data->cKsp = NULL;
  if(Kmat[currLev - 1] != NULL) {
    MPI_Comm comm;
    PetscObjectGetComm(((PetscObject)(Kmat[currLev - 1])), &comm);
    PC cPc;
    KSPCreate(comm, &(data->cKsp));
    KSPGetPC((data->cKsp), &cPc);
    MatGetVecs(Kmat[currLev - 1], &(data->cSol), &(data->cRhs));
    if(currLev > 1) {
      KSPSetType((data->cKsp), KSPFGMRES);
      KSPSetPCSide((data->cKsp), PC_RIGHT);
      setupRTG(cPc, K, (currLev - 1), da, Kmat, Pmat, tmpCvec);
    } else {
      KSPSetType((data->cKsp), KSPCG);
      KSPSetPCSide((data->cKsp), PC_LEFT);
      PCSetType(cPc, PCCHOLESKY);
      PCFactorSetShiftAmount(cPc, 1.0e-12);
      PCFactorSetShiftType(cPc, MAT_SHIFT_POSITIVE_DEFINITE);
      PCFactorSetMatSolverPackage(cPc, MATSOLVERMUMPS);
    }
    KSPSetInitialGuessNonzero(data->cKsp, PETSC_TRUE);
    KSPSetOperators(data->cKsp, Kmat[currLev - 1], Kmat[currLev - 1], SAME_PRECONDITIONER);
    KSPSetTolerances(data->cKsp, 0.1, 1.0e-12, 2.0, 1000);
    KSPDefaultConvergedSetUIRNorm(data->cKsp);
    KSPSetNormType(data->cKsp, KSP_NORM_UNPRECONDITIONED);
  }
  PCSetType(pc, PCSHELL);
  PCShellSetContext(pc, data);
  PCShellSetName(pc, "MyRTG");
  PCShellSetApply(pc, &applyRTG);
  PCShellSetDestroy(pc, &destroyRTG);
}

PetscErrorCode destroyRTG(PC pc) {
  RTGdata* data;
  PCShellGetContext(pc, (void**)(&data));
  destroySmoother(data->sData); 
  VecDestroy(&(data->res));
  if(data->cKsp != NULL) {
    KSPDestroy(&(data->cKsp));
    VecDestroy(&(data->cRhs));
    VecDestroy(&(data->cSol));
  }
  delete data;
  return 0;
}

PetscErrorCode applyRTG(PC pc, Vec in, Vec out) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)pc), &comm);
  MPI_Barrier(comm);
  int rank;
  int npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);
  std::cout<<"("<<rank<<"/"<<npes<<")"<<" Entered ApplyRTG"<<std::endl;

  RTGdata* data;
  PCShellGetContext(pc, (void**)(&data));
  VecZeroEntries(out);
  makeBoundariesConsistent(data->da, in, out, data->K);
  computeResidual(data->Kmat, out, in, data->res);
  PetscReal currNorm;
  VecNorm(data->res, NORM_2, &currNorm);
  PetscReal initNorm = currNorm;
  double tgtNorm = 0.1*initNorm;
  MPI_Barrier(comm);
  std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-1 Pass"<<std::endl;
  for(int iter = 0; iter < 1000; ++iter) {
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= tgtNorm) {
      break;
    }
    applySmoother((iter + 1), tgtNorm, currNorm, data->sData, in, out);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-2 Pass"<<std::endl;
    computeResidual(data->Kmat, out, in, data->res);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-3 Pass"<<std::endl;
    VecNorm(data->res, NORM_2, &currNorm);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-4 Pass"<<std::endl;
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= tgtNorm) {
      break;
    }
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Entering Restriction"<<std::endl;
    MPI_Barrier(comm);
    applyRestriction(data->Pmat, data->tmpCvec, data->res, data->cRhs);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Left Restriction"<<std::endl;
    MPI_Barrier(comm);
    if(data->cKsp != NULL) {
      VecZeroEntries(data->cSol);
      KSPSolve(data->cKsp, data->cRhs, data->cSol);
    }
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-6 Pass"<<std::endl;
    applyProlongation(data->Pmat, data->tmpCvec, data->cSol, data->res);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-7 Pass"<<std::endl;
    VecAXPY(out, 1.0, data->res);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-8 Pass"<<std::endl;
    computeResidual(data->Kmat, out, in, data->res);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-9 Pass"<<std::endl;
    VecNorm(data->res, NORM_2, &currNorm);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-10 Pass"<<std::endl;
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= tgtNorm) {
      break;
    }
    applySmoother((iter + 1), tgtNorm, currNorm, data->sData, in, out);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-11 Pass"<<std::endl;
    computeResidual(data->Kmat, out, in, data->res);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-12 Pass"<<std::endl;
    VecNorm(data->res, NORM_2, &currNorm);
    MPI_Barrier(comm);
    std::cout<<"("<<rank<<"/"<<npes<<")"<<"Stage-12 Pass"<<std::endl;
  }//end iter

  MPI_Barrier(comm);
  std::cout<<"("<<rank<<"/"<<npes<<")"<<" Left ApplyRTG"<<std::endl;

  return 0;
}


