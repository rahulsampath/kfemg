
#ifndef __RTG_PC__
#define __RTG_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscksp.h"
#include "petscpc.h"
#include "petscdmda.h"
#include "gmg/include/oldRtg/smoother.h"
#include <vector>

struct RTGdata {
  int K;
  DM da;
  SmootherData* sData;
  Mat Kmat;
  Mat Pmat;
  Vec tmpCvec;
  KSP cKsp;
  Vec res;
  Vec cRhs;
  Vec cSol;
};

void setupRTG(PC pc, int K, int currLev, std::vector<DM>& da, std::vector<Mat>& Kmat,
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec);

PetscErrorCode applyRTG(PC pc, Vec in, Vec out);

PetscErrorCode destroyRTG(PC pc);

#endif


