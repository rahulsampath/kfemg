
#include "gmg/include/mgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/intergrid.h"

PetscErrorCode applyMG(PC pc, Vec in, Vec out) {
  MGdata* data;
  PCShellGetContext(pc, (void**)(&data));

  int nlevels = (data->Kmat).size();
  VecZeroEntries(out);
  makeBoundariesConsistent((data->daFinest), in, out, (data->K));
  data->mgSol[data->Kmat.size() - 1] = out;
  data->mgRhs[data->Kmat.size() - 1] = in;
  applyVcycle((nlevels - 1), data);
  data->mgSol[data->Kmat.size() - 1] = NULL;
  data->mgRhs[data->Kmat.size() - 1] = NULL;

  return 0;
}

void applyVcycle(int currLev, MGdata* data) {
  if(currLev == 0) {
    KSPSolve(data->coarseSolver, data->mgRhs[currLev], data->mgSol[currLev]);
  } else {
    applySmoother(data->sData[currLev - 1], data->mgRhs[currLev], data->mgSol[currLev]);
    computeResidual(data->Kmat[currLev], data->mgSol[currLev], data->mgRhs[currLev], data->mgRes[currLev]);
    applyRestriction(data->Pmat[currLev - 1], data->tmpCvec[currLev - 1], data->mgRes[currLev], data->mgRhs[currLev - 1]);
    if(data->mgSol[currLev - 1] != NULL) {
      VecZeroEntries(data->mgSol[currLev - 1]);
      applyVcycle((currLev - 1), data);
    }
    applyProlongation(data->Pmat[currLev - 1], data->tmpCvec[currLev - 1], data->mgSol[currLev - 1], data->mgRes[currLev]);
    VecAXPY(data->mgSol[currLev], 1.0, data->mgRes[currLev]);
    applySmoother(data->sData[currLev - 1], data->mgRhs[currLev], data->mgSol[currLev]);
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



