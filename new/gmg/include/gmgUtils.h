
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
  //long double res = sin((static_cast<long double>(xFac)) * __PI__ * x);
  //long double res = pow((x - 0.5), 8) - pow(0.5, 8);
// long double res = pow((x - 0.5), 2) - pow(0.5, 2);
long double res = x;
  return res;
}

inline long double solution2D(long double x, long double y, int xFac, int yFac) {
  //long double res = solution1D(x, xFac) * solution1D(y, yFac);
  long double res = x;
  return res;
}

inline long double solutionDerivative1D(long double x, int dofX, int xFac) {
  long double res;

  if(dofX == 0) {
    res = x;
  } else if(dofX == 1) {
    res = 1.0;
  } else {
    res = 0.0;
  }

  /*
  if(dofX == 0) {
    res = solution1D(x, xFac);
  } else if(dofX > 2) {
    res = 0;
  } else if(dofX == 1) {
    res = 2 * (x - 0.5);
  } else {
    res = 2; 
  }
  */
  
  /*
  else if(dofX > 8) {
    res = 0;
  } else if(dofX == 1) {
    res = 8 * pow((x - 0.5), 7);
  } else if(dofX == 2) {
    res = 8 * 7 * pow((x - 0.5), 6);
  } else if(dofX == 3) {
    res = 8 * 7 * 6 * pow((x - 0.5), 5);
  } else if(dofX == 4) {
    res = 8 * 7 * 6 * 5 * pow((x - 0.5), 4);
  } else if(dofX == 5) {
    res = 8 * 7 * 6 * 5 * 4 * pow((x - 0.5), 3);
  } else if(dofX == 6) {
    res = 8 * 7 * 6 * 5 * 4 * 3 * pow((x - 0.5), 2);
  } else if(dofX == 7) {
    res = 8 * 7 * 6 * 5 * 4 * 3 * 2 * pow((x - 0.5), 1);
  } else {
    res = 8 * 7 * 6 * 5 * 4 * 3 * 2;
  }
  */

  /*
     res = myIntPow(((static_cast<long double>(xFac)) * __PI__), dofX);
     if(((dofX/2)%2) != 0) {
     res *= -1;
     }
     if((dofX%2) == 0) {
     res *= sin((static_cast<long double>(xFac)) * __PI__ * x);
     } else {
     res *= cos((static_cast<long double>(xFac)) * __PI__ * x);
     }
     */

  return res;
}

inline long double solutionDerivative2D(long double x, long double y, int dofX, int dofY, int xFac, int yFac) {
//  long double res = solutionDerivative1D(x, dofX, xFac) * solutionDerivative1D(y, dofY, yFac);
  long double res;
  if(dofY == 0) {
    res = solutionDerivative1D(x, dofX, xFac);
  } else {
    res = 0.0;
  }
  return res;
}

inline long double force1D(long double x, int xFac) {
  long double res = -solutionDerivative1D(x, 2, xFac);
  return res;
}

inline long double force2D(long double x, long double y, int xFac, int yFac) {
  long double res = -(solutionDerivative2D(x, y, 2, 0, xFac, yFac) + solutionDerivative2D(x, y, 0, 2, xFac, yFac));
  return res;
}

void computeRHS(DM da, std::vector<long long int>& coeffs, const int K, Vec rhs);

long double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K);

void setBoundaries(DM da, Vec vec);

void setSolution(DM da, Vec vec, const int K);

void chkBoundaries(DM da, Vec vec);

void computeKmat(Mat Kmat, DM da, std::vector<std::vector<long double> >& elemMat);

void dirichletMatrixCorrection(Mat Kmat, DM da);

#endif



