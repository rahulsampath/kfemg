
#include "gmg/include/mgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/intergrid.h"

void setupMG(PC pc, int K, int currLev, std::vector<DM>& da, std::vector<Mat>& Kmat,
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec) {
  MGdata* data = new MGdata; 
  PCSetType(pc, PCSHELL);
  PCShellSetContext(pc, &data);
  PCShellSetName(pc, "MyVcycle");
  PCShellSetApply(pc, &applyMG);
  PCShellSetDestroy(pc, &destroyMG);
}

PetscErrorCode applyMG(PC pc, Vec in, Vec out) {
  MGdata* data;
  PCShellGetContext(pc, (void**)(&data));

}

PetscErrorCode destroyMG(PC pc) {
  MGdata* data;
  PCShellGetContext(pc, (void**)(&data));
  delete data;
}


