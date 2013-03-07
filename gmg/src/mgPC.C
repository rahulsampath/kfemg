
#include "gmg/include/mgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/intergrid.h"

#ifdef DEBUG
#include <cassert>
#endif

void setupMG(PC pc, int K, int currLev, std::vector<DM>& da, std::vector<Mat>& Kmat,
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec) {
#ifdef DEBUG
  assert(currLev > 0);
#endif
  MGdata* data = new MGdata; 
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
      setupMG(cPc, K, (currLev - 1), da, Kmat, Pmat, tmpCvec);
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
  PCShellSetContext(pc, &data);
  PCShellSetName(pc, "MyVcycle");
  PCShellSetApply(pc, &applyMG);
  PCShellSetDestroy(pc, &destroyMG);
}

PetscErrorCode destroyMG(PC pc) {
  MGdata* data;
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

PetscErrorCode applyMG(PC pc, Vec in, Vec out) {
  MGdata* data;
  PCShellGetContext(pc, (void**)(&data));
  VecZeroEntries(out);
  makeBoundariesConsistent(data->da, in, out, data->K);
  computeResidual(data->Kmat, out, in, data->res);
  PetscReal currNorm;
  VecNorm(data->res, NORM_2, &currNorm);
  PetscReal initNorm = currNorm;
  for(int iter = 0; iter < 1000; ++iter) {
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= (0.5*initNorm)) {
      break;
    }
    applySmoother((iter + 1), (0.5*initNorm), currNorm, data->sData, in, out);
    computeResidual(data->Kmat, out, in, data->res);
    VecNorm(data->res, NORM_2, &currNorm);
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= (0.5*initNorm)) {
      break;
    }
    applyRestriction(data->Pmat, data->tmpCvec, data->res, data->cRhs);
    if(data->cKsp != NULL) {
      VecZeroEntries(data->cSol);
      KSPSolve(data->cKsp, data->cRhs, data->cSol);
    }
    applyProlongation(data->Pmat, data->tmpCvec, data->cSol, data->res);
    VecAXPY(out, 1.0, data->res);
    computeResidual(data->Kmat, out, in, data->res);
    VecNorm(data->res, NORM_2, &currNorm);
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= (0.5*initNorm)) {
      break;
    }
    applySmoother((iter + 1), (0.5*initNorm), currNorm, data->sData, in, out);
    computeResidual(data->Kmat, out, in, data->res);
    VecNorm(data->res, NORM_2, &currNorm);
  }//end iter
  return 0;
}


