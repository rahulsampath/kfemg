
#ifndef __MMS__
#define __MMS__

#include "gmg/include/exactSolution.i"

void setSolution(DM da, Vec vec, const int K);

void setBoundaries(DM da, Vec vec, const int K);

void computeRHS(DM da, std::vector<long long int>& coeffs, const int K, Vec rhs);

long double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K);

inline long double force1D(long double x) {
  long double res = -solutionDerivative1D(x, 2);
  return res;
}

inline long double force2D(long double x, long double y) {
  long double res = -( solutionDerivative2D(x, y, 2, 0) + solutionDerivative2D(x, y, 0, 2) );
  return res;
}

inline long double force3D(long double x, long double y, long double z) {
  long double res = -( solutionDerivative3D(x, y, z, 2, 0, 0) + 
      solutionDerivative3D(x, y, z, 0, 2, 0) + 
      solutionDerivative3D(x, y, z, 0, 0, 2) );
  return res;
}

#endif


