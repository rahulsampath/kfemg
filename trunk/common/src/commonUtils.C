
#include <cstdio>
#include <cstring>
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include "common/include/commonUtils.h"

void createPoisson3DelementMatrix(unsigned int K, std::vector<long long int> & coeffs, 
    long double hz, long double hy, long double hx, std::vector<std::vector<long double> >& mat, bool print) {
  unsigned int matSz = 8*(K + 1)*(K + 1)*(K + 1);
  if(print) {
    std::cout<<"ElemMatSize = "<<matSz<<std::endl;
  }
  mat.resize(matSz);
  for(unsigned int i = 0; i < matSz; ++i) {
    (mat[i]).resize(matSz);
  }//end i

  unsigned int numGaussPts = (2*K) + 2;
  if(print) {
    std::cout<<"NumGaussPtsPerDim = "<<numGaussPts<<std::endl;
  }
  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
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
                                                              coeffs, gPt[gZ], gPt[gY], gPt[gX], 1, 0, 0, hz, hy, hx) *
                                        eval3DshFnGderivative(cNodeZ, cNodeY, cNodeX, cDofZ, cDofY, cDofX, K,
                                          coeffs, gPt[gZ], gPt[gY], gPt[gX], 1, 0, 0, hz, hy, hx) ) + 
                                      ( eval3DshFnGderivative(rNodeZ, rNodeY, rNodeX, rDofZ, rDofY, rDofX, K,
                                                              coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 1, 0, hz, hy, hx) * 
                                        eval3DshFnGderivative(cNodeZ, cNodeY, cNodeX, cDofZ, cDofY, cDofX, K,
                                          coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 1, 0, hz, hy, hx) ) +
                                      ( eval3DshFnGderivative(rNodeZ, rNodeY, rNodeX, rDofZ, rDofY, rDofX, K,
                                                              coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 0, 1, hz, hy, hx) * 
                                        eval3DshFnGderivative(cNodeZ, cNodeY, cNodeX, cDofZ, cDofY, cDofX, K,
                                          coeffs, gPt[gZ], gPt[gY], gPt[gX], 0, 0, 1, hz, hy, hx) ) ) );
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
      mat[i][j] *= (hx*hy*hz/8.0L);
    }//end j
  }//end i
}

void createPoisson2DelementMatrix(unsigned int K, std::vector<long long int> & coeffs, 
    long double hy, long double hx, std::vector<std::vector<long double> >& mat, bool print) {
  unsigned int matSz = 4*(K + 1)*(K + 1);
  if(print) {
    std::cout<<"ElemMatSize = "<<matSz<<std::endl;
  }
  mat.resize(matSz);
  for(unsigned int i = 0; i < matSz; ++i) {
    (mat[i]).resize(matSz);
  }//end i

  unsigned int numGaussPts = (2*K) + 2;
  if(print) {
    std::cout<<"NumGaussPtsPerDim = "<<numGaussPts<<std::endl;
  }
  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  for(unsigned int rNodeY = 0, r = 0; rNodeY < 2; ++rNodeY) {
    for(unsigned int rNodeX = 0; rNodeX < 2; ++rNodeX) {
      for(unsigned int rDofY = 0; rDofY <= K; ++rDofY) {
        for(unsigned int rDofX = 0; rDofX <= K; ++rDofX, ++r) {
          for(unsigned int cNodeY = 0, c = 0; cNodeY < 2; ++cNodeY) {
            for(unsigned int cNodeX = 0; cNodeX < 2; ++cNodeX) {
              for(unsigned int cDofY = 0; cDofY <= K; ++cDofY) {
                for(unsigned int cDofX = 0; cDofX <= K; ++cDofX, ++c) {
                  mat[r][c] = 0.0;
                  for(unsigned int gY = 0; gY < numGaussPts; ++gY) {
                    for(unsigned int gX = 0; gX < numGaussPts; ++gX) {
                      mat[r][c] += ( gWt[gY] * gWt[gX] * (
                            ( eval2DshFnGderivative(rNodeY, rNodeX, rDofY, rDofX, K,
                                                    coeffs, gPt[gY], gPt[gX], 1, 0, hy, hx) * 
                              eval2DshFnGderivative(cNodeY, cNodeX, cDofY, cDofX, K,
                                coeffs, gPt[gY], gPt[gX], 1, 0, hy, hx) ) +
                            ( eval2DshFnGderivative(rNodeY, rNodeX, rDofY, rDofX, K,
                                                    coeffs, gPt[gY], gPt[gX], 0, 1, hy, hx) * 
                              eval2DshFnGderivative(cNodeY, cNodeX, cDofY, cDofX, K,
                                coeffs, gPt[gY], gPt[gX], 0, 1, hy, hx) ) ) );
                    }//end gX
                  }//end gY
                }//end cDofX
              }//end cDofY
            }//end cNodeX
          }//end cNodeY
        }//end rDofX
      }//end rDofY
    }//end rNodeX
  }//end rNodeY

  for(unsigned int i = 0; i < matSz; ++i) {
    for(unsigned int j = 0; j < matSz; ++j) {
      mat[i][j] *= (hx*hy/4.0L);
    }//end j
  }//end i
}

void createPoisson1DelementMatrix(unsigned int K, std::vector<long long int> & coeffs,
    long double hx, std::vector<std::vector<long double> >& mat, bool print) {
  unsigned int matSz = 2*(K + 1);
  if(print) {
    std::cout<<"ElemMatSize = "<<matSz<<std::endl;
  }
  mat.resize(matSz);
  for(unsigned int i = 0; i < matSz; ++i) {
    (mat[i]).resize(matSz);
  }//end i

  unsigned int numGaussPts = (2*K) + 2;
  if(print) {
    std::cout<<"NumGaussPtsPerDim = "<<numGaussPts<<std::endl;
  }
  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  for(unsigned int rNode = 0, r = 0; rNode < 2; ++rNode) {
    for(unsigned int rDof = 0; rDof <= K; ++rDof, ++r) {
      for(unsigned int cNode = 0, c = 0; cNode < 2; ++cNode) {
        for(unsigned int cDof = 0; cDof <= K; ++cDof, ++c) {
          mat[r][c] = 0.0;
          for(unsigned int g = 0; g < numGaussPts; ++g) {
            mat[r][c] += ( gWt[g] * eval1DshFnGderivative(rNode, rDof, K, coeffs, gPt[g], 1, hx) *
                eval1DshFnGderivative(cNode, cDof, K, coeffs, gPt[g], 1, hx) );
          }//end g
        }//end cDof
      }//end cNode
    }//end rDof
  }//end rNode

  for(unsigned int i = 0; i < matSz; ++i) {
    for(unsigned int j = 0; j < matSz; ++j) {
      mat[i][j] *= (hx/2.0L);
    }//end j
  }//end i
}

long double eval3DshFnGderivative(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, long double zi, long double yi, long double xi,
    int zl, int yl, int xl, long double hz, long double hy, long double hx) {

  long double result = ( std::pow((2.0L/hz), zl) * std::pow((2.0L/hy), yl) * std::pow((2.0L/hx), xl) * 
      eval1DshFnLderivative(zNodeId, zDofId, K, coeffs, zi, zl) *
      eval1DshFnLderivative(yNodeId, yDofId, K, coeffs, yi, yl) *
      eval1DshFnLderivative(xNodeId, xDofId, K, coeffs, xi, xl) );

  return result;
}

long double eval2DshFnGderivative(unsigned int yNodeId, unsigned int xNodeId, unsigned int yDofId, 
    unsigned int xDofId, unsigned int K, std::vector<long long int> & coeffs, long double yi,
    long double xi, int yl, int xl, long double hy, long double hx) {

  long double result = ( std::pow((2.0L/hy), yl) * std::pow((2.0L/hx), xl) * 
      eval1DshFnLderivative(yNodeId, yDofId, K, coeffs, yi, yl) *
      eval1DshFnLderivative(xNodeId, xDofId, K, coeffs, xi, xl) );

  return result;
}

long double eval1DshFnGderivative(unsigned int xNodeId, unsigned int xDofId, unsigned int K, 
    std::vector<long long int> & coeffs, long double xi, int xl, long double hx) {

  long double result = ( std::pow((2.0L/hx), xl) * eval1DshFnLderivative(xNodeId, xDofId, K, coeffs, xi, xl) );

  return result;
}

long double eval3DshFn(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, long double zi, long double yi, long double xi) {

  long double result = ( eval1DshFn(zNodeId, zDofId, K, coeffs, zi) *
      eval1DshFn(yNodeId, yDofId, K, coeffs, yi) *
      eval1DshFn(xNodeId, xDofId, K, coeffs, xi) );

  return result;
}

long double eval2DshFn(unsigned int yNodeId, unsigned int xNodeId, unsigned int yDofId, unsigned int xDofId, 
    unsigned int K, std::vector<long long int> & coeffs, long double yi, long double xi) {

  long double result = ( eval1DshFn(yNodeId, yDofId, K, coeffs, yi) *
      eval1DshFn(xNodeId, xDofId, K, coeffs, xi) );

  return result;
}

long double eval1DshFnLderivative(unsigned int nodeId, unsigned int dofId, unsigned int K,
    std::vector<long long int> & coeffs, long double xi, unsigned int l) {
  assert(nodeId < 2);
  assert(dofId <= K);
  assert(K <= 10);
  //xi is the coordinate in the reference element.
  assert(xi >= -1.0L);
  assert(xi <= 1.0L);
  assert( (coeffs.size()) == (8*(K + 1)*(K + 1)) );

  unsigned int P = (2*K) + 1;

  long long int* coeffArr = &(coeffs[2*(P + 1)*((nodeId*(K + 1)) + dofId)]);

  long double result = 0.0;

  for(unsigned int i = 0; i <= P; ++i) {
    long double num = coeffArr[2*i];
    long double den = coeffArr[(2*i) + 1];
    long double c = num/den;
    result += (c*powDerivative(xi, i, l));    
  }//end i

  return result;
}

long double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, long double xi) {
  assert(nodeId < 2);
  assert(dofId <= K);
  assert(K <= 10);
  //xi is the coordinate in the reference element.
  assert(xi >= -1.0L);
  assert(xi <= 1.0L);
  assert( (coeffs.size()) == (8*(K + 1)*(K + 1)) );

  unsigned int P = (2*K) + 1;

  long long int* coeffArr = &(coeffs[2*(P + 1)*((nodeId*(K + 1)) + dofId)]);

  long double result = 0.0;

  for(int i = 0; i <= P; ++i) {
    long double num = coeffArr[2*i];
    long double den = coeffArr[(2*i) + 1];
    long double c = num/den;
    result += (c*std::pow(xi, i));    
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

long double powDerivative(long double x, unsigned int i, unsigned int l) {
  long double result;

  if(l > i) {
    result = 0.0;
  } else if(l == i) {
    result = static_cast<long double>(factorial(l));
  } else {
    int p = (i - l);
    result = std::pow(x, p);
    while(p < i) {
      result *= (static_cast<long double>(p + 1));
      ++p;
    }
  }

  return result;
}

bool softEquals(long double a, long double b) {
  long double diff = a - b;
  return ((fabs(diff)) < 1.0e-15L);
}

long double legendrePoly(int n, long double x) {
  long double result;
  if(n == 0) {
    result = 1.0;
  } else if(n == 1) {
    result = x;
  } else {
    int m = n - 1;
    result = ( ((static_cast<long double>((2*m) + 1))*x*legendrePoly(m, x)) -
        ((static_cast<long double>(m))*legendrePoly((m - 1), x)) )/(static_cast<long double>(m + 1));
  }
  return result;
}

long double legendrePolyPrime(int n, long double x) {
  long double result;
  if(n == 0) {
    result = 0.0;
  } else if(n == 1) {
    result = 1.0;
  } else {
    int m = n - 1;
    result = ( ((static_cast<long double>((2*m) + 1))*legendrePoly(m, x)) +
        ((static_cast<long double>((2*m) + 1))*x*legendrePolyPrime(m, x)) - 
        ((static_cast<long double>(m))*legendrePolyPrime((m - 1), x)) )/(static_cast<long double>(m + 1));
  }
  return result;
}

long double gaussWeight(int n, long double x) {
  long double polyPrime = legendrePolyPrime(n, x);
  return (2.0L/((1.0L - (x*x))*polyPrime*polyPrime));
}

int getDofsPerNode(int dim, int K) { 
  int dofsPerNode = (K + 1);
  assert(dim > 0);
  assert(dim <= 3);
  if(dim > 1) {
    dofsPerNode *= (K + 1);
  }
  if(dim > 2) {
    dofsPerNode *= (K + 1);
  }
  return dofsPerNode;
}

long long int myIntPow(long long int base, unsigned int exp) {
  long long int res = 1;
  for(int i = 0; i < exp; ++i) {
    res *= base;
  }//end i
  return res;
}

void initFactorials(std::vector<unsigned long long int>& fac) { 
  //REMARK: 21! requires more than 64 bits.
  fac.resize(21);
  fac[0] = 1ULL;
  fac[1] = 1ULL;
  fac[2] = 2ULL;
  fac[3] = 6ULL;
  fac[4] = 24ULL;
  fac[5] = 120ULL;
  fac[6] = 720ULL;
  fac[7] = 5040ULL;
  fac[8] = 40320ULL;
  fac[9] = 362880ULL;
  fac[10] = 3628800ULL;
  fac[11] = 39916800ULL;
  fac[12] = 479001600ULL;
  fac[13] = 6227020800ULL;
  fac[14] = 87178291200ULL;
  fac[15] = 1307674368000ULL;
  fac[16] = 20922789888000ULL;
  fac[17] = 355687428096000ULL;
  fac[18] = 6402373705728000ULL;
  fac[19] = 121645100408832000ULL;
  fac[20] = 2432902008176640000ULL;
} 



