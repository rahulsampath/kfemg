
#ifndef __EXACT_SOLUTION__
#define __EXACT_SOLUTION__

#include <cmath>

inline long double solution1D(long double x) {
  long double xStar = 0.5;
  long double sigma = 0.3;
  long double val = (x - xStar)/sigma;
  long double res = exp(val*val);
  return res;
}

inline long double solutionDerivative1D(long double x, int dofX) {
  long double xStar = 0.5;
  long double sigma = 0.3;
  long double res;
  if(dofX == 0) {
    res = solution1D(x);
  } else if(dofX == 1) {
    res = 2.0L*(x - xStar)*(solution1D(x))/(sigma*sigma);
  } else {
    res = (2.0L/(sigma*sigma))*( ((x - xStar)*(solutionDerivative1D(x, (dofX - 1)))) +
        ((dofX - 1)*(solutionDerivative1D(x, (dofX - 2)))) );
  }
  return res;
}

inline long double solution2D(long double x, long double y) {
  long double res = solution1D(x) * solution1D(y);
  return res;
}

inline long double solution3D(long double x, long double y, long double z) {
  long double res = solution1D(x) * solution1D(y) * solution1D(z);
  return res;
}

inline long double solutionDerivative2D(long double x, long double y, int dofX, int dofY) {
  long double res = solutionDerivative1D(x, dofX) * solutionDerivative1D(y, dofY);
  return res;
}

inline long double solutionDerivative3D(long double x, long double y, long double z, int dofX, int dofY, int dofZ) {
  long double res = solutionDerivative1D(x, dofX) * solutionDerivative1D(y, dofY) * solutionDerivative1D(z, dofZ);
  return res;
}

#endif

