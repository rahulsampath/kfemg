
#include "gmg/include/newSmoother.h"
#include "gmg/include/gmgUtils.h"

void setupNewSmoother(NewSmootherData* data, int K, int currLev,
    std::vector<std::vector<DM> >& da, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)(Kmat[K][currLev])), &comm);
  data->K = K;
  data->Kmat = Kmat[K][currLev];
  data->da = da[K][currLev];
  MatGetVecs((data->Kmat), PETSC_NULL, &(data->res));
}

void destroyNewSmoother(NewSmootherData* data) {
}

void applyNewSmoother(int maxIters, double tgtNorm, double currNorm,
    NewSmootherData* data, Vec in, Vec out) {
}


