
#ifndef __LS__
#define __LS__

#include "petsc.h"
#include "petscmat.h"
#include "petscvec.h"

struct LSdata {
  Mat Kmat;
  Vec w1;
  Vec w2;
};

void setupLS(LSdata* data, Mat Kmat);

void destroyLS(LSdata* data);

double applyLS(LSdata* data, Vec g, Vec v1, Vec v2, double a[2]); 

#endif

