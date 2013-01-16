
#include <vector>
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, 
    std::vector<Vec>& tmpCvec, std::vector<KSP>& smoother, KSP coarseSolver,
    std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes) {
  if(currLev == 0) {
    KSPSolve(coarseSolver, mgRhs[currLev], mgSol[currLev]);
  } else {
    KSPSolve(smoother[currLev - 1], mgRhs[currLev], mgSol[currLev]);
    computeResidual(Kmat[currLev], mgSol[currLev], mgRhs[currLev], mgRes[currLev]);
    applyRestriction(Pmat[currLev - 1], tmpCvec[currLev - 1], mgRes[currLev], mgRhs[currLev - 1]);
    if(mgSol[currLev - 1] != NULL) {
      VecZeroEntries(mgSol[currLev - 1]);
      applyVcycle((currLev - 1), Kmat, Pmat, tmpCvec, smoother,
          coarseSolver, mgSol, mgRhs, mgRes);
    }
    applyProlongation(Pmat[currLev - 1], tmpCvec[currLev - 1], mgSol[currLev - 1], mgRes[currLev]);
    VecAXPY(mgSol[currLev], 1.0, mgRes[currLev]);
    KSPSolve(smoother[currLev - 1], mgRhs[currLev], mgSol[currLev]);
  }
}

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes) {
  mgSol.resize(Kmat.size(), NULL);
  mgRhs.resize(Kmat.size(), NULL);
  mgRes.resize(Kmat.size(), NULL);
  for(size_t i = 0; i < (Kmat.size() - 1); ++i) {
    if(Kmat[i] != NULL) {
      MatGetVecs(Kmat[i], &(mgSol[i]), &(mgRhs[i]));
      VecDuplicate(mgRhs[i], &(mgRes[i]));
    }
  }//end i
  MatGetVecs(Kmat[Kmat.size() - 1], NULL, &(mgRes[Kmat.size() - 1]));
}

void destroyKSP(std::vector<KSP>& ksp) {
  for(size_t i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(&(ksp[i]));
    }
  }//end i
  ksp.clear();
}



