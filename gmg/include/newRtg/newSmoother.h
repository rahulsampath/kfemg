
#ifndef __NEW_SMOOTHER__
#define __NEW_SMOOTHER__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include "gmg/include/loa.h"
#include "gmg/include/ls.h"

struct NewSmootherData {
  int K;
  DM daL;
  DM daH;
  Mat Kmat;
  KSP ksp1;
  KSP ksp2;
  KSP ksp3;
  Vec res;
  Vec low;
  Vec high;
  Vec loaRhs;
  Vec loaSol;
  LOAdata* loa;
  LSdata* ls;
};

void setupNewSmoother(NewSmootherData* data, int K, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<long long int> >& coeffs, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec);

void destroyNewSmoother(NewSmootherData* data);

void applyNewSmoother(int maxIters, double tgtNorm, double currNorm,
    NewSmootherData* data, Vec in, Vec out);

#endif

