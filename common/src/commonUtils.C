
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include "common/include/commonUtils.h"

void createPoisson3DelementMatrix(unsigned int K, std::vector<long long int> & coeffs, 
    double hz, double hy, double hx, std::vector<std::vector<double> >& mat) {
  unsigned int matSz = 8*(K + 1)*(K + 1)*(K + 1);
  mat.resize(matSz);
  for(unsigned int i = 0; i < matSz; ++i) {
    (mat[i]).resize(matSz);
  }//end i

  unsigned int numGaussPts = K + 1;
  std::vector<double> gPt(numGaussPts);
  std::vector<double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  for(unsigned int rNodeZ = 0, r = 0; rNodeZ < 2; ++rNodeZ) {
    for(unsigned int rNodeY = 0; rNodeY < 2; ++rNodeY) {
      for(unsigned int rNodeX = 0; rNodeX < 2; ++rNodeX) {
        for(unsigned int rDofZ = 0; rDofZ <= K; ++rDofZ) {
          for(unsigned int rDofY = 0; rDofY <= K; ++rDofY) {
            for(unsigned int rDofX = 0; rDofX <= K; ++rDofX, ++r) {
              for(unsigned int cNodeZ = 0, c = 0; cNodeZ < 2; ++cNodeZ) {
                for(unsigned int cNodeY = 0; cNodeY < 2; ++cNodeY) {
                  for(unsigned int cNodeX = 0; cNodeX < 2; ++cNodeX) {
                    for(unsigned int cDofZ = 0; cDofZ <= K; ++cDofZ) {
                      for(unsigned int cDofY = 0; cDofY <= K; ++cDofY) {
                        for(unsigned int cDofX = 0; cDofX <= K; ++cDofX, ++c) {
                          mat[r][c] = 0.0;
                          for(unsigned int gZ = 0; gZ < numGaussPts; ++gZ) {
                            for(unsigned int gY = 0; gY < numGaussPts; ++gY) {
                              for(unsigned int gX = 0; gX < numGaussPts; ++gX) {
                                mat[r][c] += ( gWt[gZ] * gWt[gY] * gWt[gX] * (
                                      ( eval3DshFnGderivative(rNodeZ, rNodeY, rNodeX, rDofZ, rDofY, rDofX, K,
                                                              coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 0, 1, hz, hy, hx) * 
                                        eval3DshFnGderivative(cNodeZ, cNodeY, cNodeX, cDofZ, cDofY, cDofX, K,
                                          coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 0, 1, hz, hy, hx) ) +
                                      ( eval3DshFnGderivative(rNodeZ, rNodeY, rNodeX, rDofZ, rDofY, rDofX, K,
                                                              coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 1, 0, hz, hy, hx) * 
                                        eval3DshFnGderivative(cNodeZ, cNodeY, cNodeX, cDofZ, cDofY, cDofX, K,
                                          coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 1, 0, hz, hy, hx) ) +
                                      ( eval3DshFnGderivative(rNodeZ, rNodeY, rNodeX, rDofZ, rDofY, rDofX, K,
                                                              coeffs, gPt[gZ], gPt[gY], gPt[gX], 1, 0, 0, hz, hy, hx) *
                                        eval3DshFnGderivative(cNodeZ, cNodeY, cNodeX, cDofZ, cDofY, cDofX, K,
                                          coeffs, gPt[gZ], gPt[gY], gPt[gX], 1, 0, 0, hz, hy, hx) ) 
                                      ) );
                              }//end gX
                            }//end gY
                          }//end gZ
                        }//end cDofX
                      }//end cDofY
                    }//end cDofZ
                  }//end cNodeX
                }//end cNodeY
              }//end cNodeZ
            }//end rDofX
          }//end rDofY
        }//end rDofZ
      }//end rNodeX
    }//end rNodeY
  }//end rNodeZ

  for(unsigned int i = 0; i < matSz; ++i) {
    for(unsigned int j = 0; j < matSz; ++j) {
      mat[i][j] *= (hx*hy*hz/8.0);
    }//end j
  }//end i
}

void createPoisson1DelementMatrix(unsigned int K, std::vector<long long int> & coeffs,
    double hx, std::vector<std::vector<double> >& mat) {
  unsigned int matSz = 2*(K + 1);
  mat.resize(matSz);
  for(unsigned int i = 0; i < matSz; ++i) {
    (mat[i]).resize(matSz);
  }//end i

  unsigned int numGaussPts = K + 1;
  std::vector<double> gPt(numGaussPts);
  std::vector<double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  for(unsigned int rNode = 0, r = 0; rNode < 2; ++rNode) {
    for(unsigned int rDof = 0; rDof <= K; ++rDof, ++r) {
      for(unsigned int cNode = 0, c = 0; cNode < 2; ++cNode) {
        for(unsigned int cDof = 0; cDof <= K; ++cDof, ++c) {
          mat[r][c] = 0.0;
          for(unsigned int g = 0; g < numGaussPts; ++g) {
            mat[r][c] += ( gWt[g] * eval1DshFnLderivative(rNode, rDof, K, coeffs, gPt[g], 1) *
                eval1DshFnLderivative(cNode, cDof, K, coeffs, gPt[g], 1) );
          }//end g
        }//end cDof
      }//end cNode
    }//end rDof
  }//end rNode

  for(unsigned int i = 0; i < matSz; ++i) {
    for(unsigned int j = 0; j < matSz; ++j) {
      mat[i][j] *= (2.0/hx);
    }//end j
  }//end i
}

double eval3DshFnGderivative(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, double zi, double yi, double xi,
    unsigned int zl, unsigned int yl, unsigned int xl, double hz, double hy, double hx) {

  double result = ( pow((2.0/hz), zl) * pow((2.0/hy), yl) * pow((2.0/hx), xl) * 
      eval1DshFnLderivative(zNodeId, zDofId, K, coeffs, zi, zl) *
      eval1DshFnLderivative(yNodeId, yDofId, K, coeffs, yi, yl) *
      eval1DshFnLderivative(xNodeId, xDofId, K, coeffs, xi, xl) );

  return result;
}

double eval3DshFn(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, double zi, double yi, double xi) {

  double result = ( eval1DshFn(zNodeId, zDofId, K, coeffs, zi) *
      eval1DshFn(yNodeId, yDofId, K, coeffs, yi) *
      eval1DshFn(xNodeId, xDofId, K, coeffs, xi) );

  return result;
}

double eval1DshFnLderivative(unsigned int nodeId, unsigned int dofId, unsigned int K,
    std::vector<long long int> & coeffs, double xi, unsigned int l) {
  assert(nodeId < 2);
  assert(dofId <= K);
  assert(K <= 10);
  //xi is the coordinate in the reference element.
  assert(xi >= -1.0);
  assert(xi <= 1.0);
  assert( (coeffs.size()) == (8*(K + 1)*(K + 1)) );

  unsigned int P = (2*K) + 1;

  long long int* coeffArr = &(coeffs[2*(P + 1)*((nodeId*(K + 1)) + dofId)]);

  double result = 0.0;

  for(unsigned int i = 0; i <= P; ++i) {
    double num = coeffArr[2*i];
    double den = coeffArr[(2*i) + 1];
    double c = num/den;
    result += (c*powDerivative(xi, i, l));    
  }//end i

  return result;
}

double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, double xi) {
  assert(nodeId < 2);
  assert(dofId <= K);
  assert(K <= 10);
  //xi is the coordinate in the reference element.
  assert(xi >= -1.0);
  assert(xi <= 1.0);
  assert( (coeffs.size()) == (8*(K + 1)*(K + 1)) );

  unsigned int P = (2*K) + 1;

  long long int* coeffArr = &(coeffs[2*(P + 1)*((nodeId*(K + 1)) + dofId)]);

  double result = 0.0;

  for(int i = 0; i <= P; ++i) {
    double num = coeffArr[2*i];
    double den = coeffArr[(2*i) + 1];
    double c = num/den;
    result += (c*pow(xi, i));    
  }//end i

  return result;
}

void read1DshapeFnCoeffs(unsigned int K, std::vector<long long int> & coeffs) {
  char fname[256];
  sprintf(fname, "../../common/ShFnCoeffs1D/C%uShFnCoeffs1D.txt", K);

  FILE *fp = fopen(fname, "r"); 

  assert(fp != NULL);

  unsigned int numCoeffs = 4*(K + 1)*(K + 1);

  coeffs.resize(2*numCoeffs);
  for(unsigned int i = 0; i < (2*numCoeffs); ++i) {
    fscanf(fp, "%lld", &(coeffs[i]));
  }//end i 

  fclose(fp);
}

double powDerivative(double x, unsigned int i, unsigned int l) {
  double result;

  if(l > i) {
    result = 0.0;
  } else if(l == i) {
    result = static_cast<double>(factorial(l));
  } else {
    unsigned int p = (i - l);
    result = pow(x, (static_cast<int>(p)));
    while(p < i) {
      result *= (static_cast<double>(p + 1));
      ++p;
    }
  }

  return result;
}

bool softEquals(double a, double b) {
  return ((fabs(a - b)) < 1.0e-14);
}

double legendrePoly(int n, double x) {
  double result;
  if(n == 0) {
    result = 1.0;
  } else if(n == 1) {
    result = x;
  } else {
    int m = n - 1;
    result = ( ((static_cast<double>((2*m) + 1))*x*legendrePoly(m, x)) -
        ((static_cast<double>(m))*legendrePoly((m - 1), x)) )/(static_cast<double>(m + 1));
  }
  return result;
}

double legendrePolyPrime(int n, double x) {
  double result;
  if(n == 0) {
    result = 0.0;
  } else if(n == 1) {
    result = 1.0;
  } else {
    int m = n - 1;
    result = ( ((static_cast<double>((2*m) + 1))*legendrePoly(m, x)) +
        ((static_cast<double>((2*m) + 1))*x*legendrePolyPrime(m, x)) - 
        ((static_cast<double>(m))*legendrePolyPrime((m - 1), x)) )/(static_cast<double>(m + 1));
  }
  return result;
}

double gaussWeight(int n, double x) {
  double polyPrime = legendrePolyPrime(n, x);
  return (2.0/((1.0 - (x*x))*polyPrime*polyPrime));
}


