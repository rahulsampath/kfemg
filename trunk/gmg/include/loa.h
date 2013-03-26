
#ifndef __LOA__
#define __LOA__

#include "petsc.h"
#include "petscvec.h"
#include "petscdmda.h"
#include <vector>

struct LOAdata {
  int K;
  std::vector<std::vector<long long int> >* coeffs;
  DM daL;
  DM daH;
};

struct PointAndVal {
  int x;
  int y;
  int z;
  double v;
  bool operator < (PointAndVal<DIM>& other) {
    return (v < (other.v));
  }
};

void setupLOA(LOAdata* data, int K, DM daL, DM daH, 
    std::vector<std::vector<long long int> >& coeffs);

void destroyLOA(LOAdata* data);

void applyLOA(LOAdata* data, Vec high, Vec low);

#endif

