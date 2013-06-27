
#include "gmg/include/static/smoother.h"
#include "gmg/include/static/mgPC.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/fd.h"
#include <cmath>

void setupSmoother(SmootherData* data, int K, int currLev, std::vector<std::vector<DM> >& da,
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
  KSPSetType(data->ksp1, KSPFGMRES);
  KSPSetPCSide(data->ksp1, PC_RIGHT);
  KSPGetPC(data->ksp1, &pc1);
  PCSetType(pc1, PCSOR);
  KSPSetInitialGuessNonzero(data->ksp1, PETSC_TRUE);
  KSPSetOperators(data->ksp1, Kmat[K][currLev], Kmat[K][currLev], SAME_PRECONDITIONER);
  KSPSetTolerances(data->ksp1, 1.0e-12, 1.0e-12, 2.0, 2);
  KSPDefaultConvergedSetUIRNorm(data->ksp1);
  KSPSetNormType(data->ksp1, KSP_NORM_UNPRECONDITIONED);
  data->ksp2 = NULL;
  data->low = NULL;
  data->high = NULL;
  data->loaRhs = NULL;
  data->loaSol = NULL;
  if(K > 0) {
    data->daL = da[K - 1][currLev];
    PC pc2;
    KSPCreate(comm, &(data->ksp2));
    KSPSetType(data->ksp2, KSPFGMRES);
    KSPSetPCSide(data->ksp2, PC_RIGHT);
    KSPGetPC(data->ksp2, &pc2);
    setupMG(pc2, (K-1), currLev, da, coeffs, Kmat, Pmat, tmpCvec); 
    KSPSetInitialGuessNonzero(data->ksp2, PETSC_TRUE);
    KSPSetOperators(data->ksp2, Kmat[K-1][currLev], Kmat[K-1][currLev], SAME_PRECONDITIONER);
    KSPSetTolerances(data->ksp2, 1.0e-12, 1.0e-12, 2.0, 1);
    KSPDefaultConvergedSetUIRNorm(data->ksp2);
    KSPSetNormType(data->ksp2, KSP_NORM_UNPRECONDITIONED);
    data->loa = new LOAdata;
    setupLOA(data->loa, K, (data->daL), (data->daH), coeffs);
    data->ls = new LSdata;
    setupLS(data->ls, Kmat[K][currLev]);
    VecDuplicate((data->res), &(data->low));
    VecDuplicate((data->res), &(data->high));
    MatGetVecs((Kmat[K-1][currLev]), &(data->loaSol), &(data->loaRhs));
  }
}

void destroySmoother(SmootherData* data) {
  KSPDestroy(&(data->ksp1));
  if((data->K) > 0) {
    KSPDestroy(&(data->ksp2));
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

void applySmoother(SmootherData* data, Vec in, Vec out) {
}



