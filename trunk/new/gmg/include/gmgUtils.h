
#ifndef __GMG_UTILS__
#define __GMG_UTILS__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include "petscpc.h"
#include <vector>
#include "mpi.h"
#include "common/include/commonUtils.h"

inline long double solution1D(long double x, int xFac) {
  long double res = sin((static_cast<long double>(xFac)) * __PI__ * x);
  return res;
}

inline long double solutionDerivative1D(long double x, int dofX, int xFac) {
  long double res = myIntPow(((static_cast<long double>(xFac)) * __PI__), dofX);
  if(((dofX/2)%2) != 0) {
    res *= -1;
  }
  if((dofX%2) == 0) {
    res *= sin((static_cast<long double>(xFac)) * __PI__ * x);
  } else {
    res *= cos((static_cast<long double>(xFac)) * __PI__ * x);
  }
  return res;
}

inline long double force1D(long double x, int xFac) {
  long double res = -solutionDerivative1D(x, 2, xFac);
  return res;
}

long double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K);

#endif



