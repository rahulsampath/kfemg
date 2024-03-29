
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
  bool operator < (PointAndVal const & other) const {
    return (v < (other.v));
  }
};

void setupLOA(LOAdata* data, int K, DM daL, DM daH, 
    std::vector<std::vector<long long int> >& coeffs);

void destroyLOA(LOAdata* data);

void applyLOA(LOAdata* data, Vec high, Vec low);

void computePstar(DM da, Vec vec, std::vector<int>& pStar);

void computeFhat(DM da, int K, std::vector<long long int>& coeffs, std::vector<int>& pStar,
    std::vector<double>& aStar, std::vector<double>& cStar, Vec out);

#endif

