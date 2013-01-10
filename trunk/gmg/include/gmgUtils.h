
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

inline long double solution1D(long double x) {
  long double res = sin(__PI__ * x);
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

inline long double solutionDerivative1D(long double x, int dofX) {
  long double res = myIntPow(__PI__, dofX);
  if(((dofX/2)%2) != 0) {
    res *= -1;
  }
  if((dofX%2) == 0) {
    res *= sin(__PI__ * x);
  } else {
    res *= cos(__PI__ * x);
  }
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

long double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K);

void computeRHS(DM da, std::vector<long long int>& coeffs, const int K, Vec rhs);

void setSolution(DM da, Vec vec, const int K);

void setBoundaries(DM da, Vec vec, const int K);

void dirichletMatrixCorrection(Mat Kmat, DM da, const int K);

void computeKmat(Mat Kmat, DM da, std::vector<std::vector<long double> >& elemMat);

#endif



