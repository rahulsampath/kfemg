
#include "gmg/include/static/mgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/intergrid.h"

#ifdef DEBUG
#include <cassert>
#endif

void setupMG(PC pc, int K, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<long long int> >& coeffs, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec) {
#ifdef DEBUG
  assert(currLev > 0);
#endif
  MGdata* data = new MGdata; 
  data->K = K;
  data->da = da[K][currLev];
  data->Kmat = Kmat[K][currLev];
  data->Pmat = Pmat[K][currLev - 1];
  data->tmpCvec = tmpCvec[K][currLev - 1];
  data->sData = new SmootherData;
  setupSmoother(data->sData, K, currLev, da, coeffs, Kmat, Pmat, tmpCvec);
  MatGetVecs(Kmat[K][currLev], PETSC_NULL, &(data->res));
  data->cKsp = NULL;
  data->cRhs = NULL;
  data->cSol = NULL;
  if(Kmat[K][currLev - 1] != NULL) {
    MPI_Comm comm;
    PetscObjectGetComm(((PetscObject)(Kmat[K][currLev - 1])), &comm);
    PC cPc;
    KSPCreate(comm, &(data->cKsp));
    KSPGetPC((data->cKsp), &cPc);
    MatGetVecs(Kmat[K][currLev - 1], &(data->cSol), &(data->cRhs));
    if(currLev > 1) {
      KSPSetType((data->cKsp), KSPFGMRES);
      KSPSetPCSide((data->cKsp), PC_RIGHT);
      setupMG(cPc, K, (currLev - 1), da, coeffs, Kmat, Pmat, tmpCvec);
    } else {
      KSPSetType((data->cKsp), KSPCG);
      KSPSetPCSide((data->cKsp), PC_LEFT);
      PCSetType(cPc, PCCHOLESKY);
      PCFactorSetShiftAmount(cPc, 1.0e-12);
      PCFactorSetShiftType(cPc, MAT_SHIFT_POSITIVE_DEFINITE);
      PCFactorSetMatSolverPackage(cPc, MATSOLVERMUMPS);
    }
    KSPSetInitialGuessNonzero(data->cKsp, PETSC_TRUE);
    KSPSetOperators(data->cKsp, Kmat[K][currLev - 1], Kmat[K][currLev - 1], SAME_PRECONDITIONER);
    KSPSetTolerances(data->cKsp, 1.0e-12, 1.0e-12, 2.0, 1);
    KSPDefaultConvergedSetUIRNorm(data->cKsp);
    KSPSetNormType(data->cKsp, KSP_NORM_UNPRECONDITIONED);
  }
  PCSetType(pc, PCSHELL);
  PCShellSetContext(pc, data);
  PCShellSetName(pc, "MyMG");
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
  applySmoother(data->sData, in, out);
  computeResidual(data->Kmat, out, in, data->res);
  applyRestriction(data->Pmat, data->tmpCvec, data->res, data->cRhs);
  if(data->cKsp != NULL) {
    VecZeroEntries(data->cSol);
    KSPSolve(data->cKsp, data->cRhs, data->cSol);
  }
  applyProlongation(data->Pmat, data->tmpCvec, data->cSol, data->res);
  VecAXPY(out, 1.0, data->res);
  applySmoother(data->sData, in, out);
  return 0;
}





