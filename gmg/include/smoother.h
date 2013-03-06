
#ifndef __SMOOTHER__
#define __SMOOTHER__

#include "petsc.h"
#include "petscvec.h"

struct SmootherData {
  Mat Kmat;
  KSP ksp1;
  KSP ksp2;
  Vec res;
};

void setupSmoother(SmootherData* data, Mat Kmat);

void destroySmoother(SmootherData* data);

void applySmoother(SmootherData* data, Vec in, Vec out);

#endif


