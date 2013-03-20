
#ifndef __NEW_RTG_PC__
#define __NEW_RTG_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscdmda.h"
#include "petscmat.h"
#include "petscksp.h"
#include "petscpc.h"
#include "gmg/include/newSmoother.h"
#include <vector>

struct NewRTGdata {
  int K;
  DM da;
  NewSmootherData* sData;
  Mat Kmat;
  Mat Pmat;
  Vec tmpCvec;
  KSP cKsp;
  Vec res;
  Vec cRhs;
  Vec cSol;
};

void setupNewRTG(PC pc, int K, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<long long int> >& coeffs, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec);

PetscErrorCode applyNewRTG(PC pc, Vec in, Vec out);

PetscErrorCode destroyNewRTG(PC pc);

#endif


