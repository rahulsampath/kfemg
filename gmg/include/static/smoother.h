
#ifndef __SMOOTHER__
#define __SMOOTHER__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include "gmg/include/loa.h"
#include "gmg/include/ls.h"

struct SmootherData {
  int K;
  DM daL;
  DM daH;
  Mat Kmat;
  KSP ksp1;
  KSP ksp2;
  Vec res;
  Vec low;
  Vec high;
  Vec loaRhs;
  Vec loaSol;
  LOAdata* loa;
  LSdata* ls;
};

void setupSmoother(SmootherData* data, int K, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<long long int> >& coeffs, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec);

void destroySmoother(SmootherData* data);

void applySmoother(SmootherData* data, Vec in, Vec out);

#endif

