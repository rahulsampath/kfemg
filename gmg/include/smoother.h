
#ifndef __SMOOTHER__
#define __SMOOTHER__

#include "petsc.h"
#include "petscvec.h"

struct SmootherData {
};

/*

struct SmootherData {
  PetscInt maxIts;
  PetscReal tol;
  Mat Kmat;
  KSP ksp1;
  KSP ksp2;
  Vec res;
};

void setupSmootherData(SmootherData* data, Mat Kmat);

void destroySmootherData(SmootherData* data);

void applySmoother(SmootherData* data, Vec in, Vec out);

*/

#endif

