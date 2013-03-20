
#ifndef __LS__
#define __LS__

#include "petsc.h"
#include "petscmat.h"
#include "petscvec.h"

struct LSdata {
  Mat Kmat;
};

void setupLS(LSdata* data, Mat Kmat);

void destroyLS(LSdata* data);

#endif

