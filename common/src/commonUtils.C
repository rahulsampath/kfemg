
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void eigenVals2x2(double mat[2][2], double val[2]) {
  double a = mat[0][0];
  double d = mat[1][1];
  double det = det2x2(mat);
  double tr = a + d;
  double tmp1 = 0.5*tr;
  double tmp2 = std::sqrt((0.25*tr*tr) - det);
  val[0] = tmp1 + tmp2;
  val[1] = tmp1 - tmp2;
}

void matMult3x3(double mat[3][3], double in[3], double out[3]) {
  for(int i = 0; i < 3; ++i) {
    out[i] = 0;
    for(int j = 0; j < 3; ++j) {
      out[i] += (mat[i][j]*in[j]);
    }//end j
  }//end i
}

void matMult2x2(double mat[2][2], double in[2], double out[2]) {
  for(int i = 0; i < 2; ++i) {
    out[i] = 0;
    for(int j = 0; j < 2; ++j) {
      out[i] += (mat[i][j]*in[j]);
    }//end j
  }//end i
}

void matInvert3x3(double mat[3][3], double matInv[3][3]) {
  double a1 = mat[0][0];
  double a2 = mat[0][1];
  double a3 = mat[0][2];
  double b1 = mat[1][0];
  double b2 = mat[1][1];
  double b3 = mat[1][2];
  double c1 = mat[2][0];
  double c2 = mat[2][1];
  double c3 = mat[2][2];
  double det = det3x3(mat);
#ifdef DEBUG
  assert(fabs(det) > 1.0e-12);
#endif
  double tmp[2][2];

  tmp[0][0] = b2;
  tmp[0][1] = b3;
  tmp[1][0] = c2;
  tmp[1][1] = c3;
  matInv[0][0] = (det2x2(tmp))/det;

  tmp[0][0] = a3;
  tmp[0][1] = a2;
  tmp[1][0] = c3;
  tmp[1][1] = c2;
  matInv[0][1] = (det2x2(tmp))/det;

  tmp[0][0] = a2;
  tmp[0][1] = a3;
  tmp[1][0] = b2;
  tmp[1][1] = b3;
  matInv[0][2] = (det2x2(tmp))/det;

  tmp[0][0] = b3;
  tmp[0][1] = b1;
  tmp[1][0] = c3;
  tmp[1][1] = c1;
  matInv[1][0] = (det2x2(tmp))/det;

  tmp[0][0] = a1;
  tmp[0][1] = a3;
  tmp[1][0] = c1;
  tmp[1][1] = c3;
  matInv[1][1] = (det2x2(tmp))/det;

  tmp[0][0] = a3;
  tmp[0][1] = a1;
  tmp[1][0] = b3;
  tmp[1][1] = b1;
  matInv[1][2] = (det2x2(tmp))/det;

  tmp[0][0] = b1;
  tmp[0][1] = b2;
  tmp[1][0] = c1;
  tmp[1][1] = c2;
  matInv[2][0] = (det2x2(tmp))/det;

  tmp[0][0] = a2;
  tmp[0][1] = a1;
  tmp[1][0] = c2;
  tmp[1][1] = c1;
  matInv[2][1] = (det2x2(tmp))/det;

  tmp[0][0] = a1;
  tmp[0][1] = a2;
  tmp[1][0] = b1;
  tmp[1][1] = b2;
  matInv[2][2] = (det2x2(tmp))/det;
}

void matInvert2x2(double mat[2][2], double matInv[2][2]) {
  double a = mat[0][0];
  double b = mat[0][1];
  double c = mat[1][0];
  double d = mat[1][1];
  double det = det2x2(mat);
#ifdef DEBUG
  assert(fabs(det) > 1.0e-12);
#endif
  matInv[0][0] = d/det;
  matInv[0][1] = -b/det;
  matInv[1][0] = -c/det;
  matInv[1][1] = a/det;
}

double det2x2(double mat[2][2]) {
  double a = mat[0][0];
  double b = mat[0][1];
  double c = mat[1][0];
  double d = mat[1][1];
  double det = (a*d) - (b*c);
  return det;
}

double det3x3(double mat[3][3]) {
  double a1 = mat[0][0];
  double a2 = mat[0][1];
  double a3 = mat[0][2];
  double b1 = mat[1][0];
  double b2 = mat[1][1];
  double b3 = mat[1][2];
  double c1 = mat[2][0];
  double c2 = mat[2][1];
  double c3 = mat[2][2];
  double det = (a1 * b2 * c3) - (a1 * b3 * c2) 
    - (a2 * b1 * c3) + (a2 * b3 * c1)
    + (a3 * b1 * c2) - (a3 * b2 * c1);
  return det;
}

void createPoisson3DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hz, long double hy, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print) {
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

  std::vector<std::vector<std::vector<long double> > > shFnDerivatives(2);
  for(unsigned int node = 0; node < 2; ++node) {
    shFnDerivatives[node].resize(K + 1);
    for(unsigned int dof = 0; dof <= K; ++dof) {
      (shFnDerivatives[node][dof]).resize(numGaussPts);
      for(unsigned int g = 0; g < numGaussPts; ++g) {
        shFnDerivatives[node][dof][g] = eval1DshFnDerivative(factorialsList, node, dof, K, coeffs, gPt[g], 1);
      }//end g
    }//end dof
  }//end node

  std::vector<std::vector<std::vector<long double> > > shFnVals(2);
  for(unsigned int node = 0; node < 2; ++node) {
    shFnVals[node].resize(K + 1);
    for(unsigned int dof = 0; dof <= K; ++dof) {
      (shFnVals[node][dof]).resize(numGaussPts);
      for(unsigned int g = 0; g < numGaussPts; ++g) {
        shFnVals[node][dof][g] = eval1DshFn(node, dof, K, coeffs, gPt[g]);
      }//end g
    }//end dof
  }//end node

  std::vector<std::vector<std::vector<long double> > > gradPhi(matSz);
  for(unsigned int nodeZ = 0, i = 0; nodeZ < 2; ++nodeZ) {
    for(unsigned int nodeY = 0; nodeY < 2; ++nodeY) {
      for(unsigned int nodeX = 0; nodeX < 2; ++nodeX) {
        for(unsigned int dofZ = 0; dofZ <= K; ++dofZ) {
          for(unsigned int dofY = 0; dofY <= K; ++dofY) {
            for(unsigned int dofX = 0; dofX <= K; ++dofX, ++i) {
              gradPhi[i].resize(numGaussPts*numGaussPts*numGaussPts);
              for(unsigned int gZ = 0, g = 0; gZ < numGaussPts; ++gZ) {
                for(unsigned int gY = 0; gY < numGaussPts; ++gY) {
                  for(unsigned int gX = 0; gX < numGaussPts; ++gX, ++g) {
                    gradPhi[i][g].resize(3);
                    gradPhi[i][g][0] = (2.0L/hx) * shFnVals[nodeZ][dofZ][gZ] * shFnVals[nodeY][dofY][gY] * shFnDerivatives[nodeX][dofX][gX];
                    gradPhi[i][g][1] = (2.0L/hy) * shFnVals[nodeZ][dofZ][gZ] * shFnDerivatives[nodeY][dofY][gY] * shFnVals[nodeX][dofX][gX];
                    gradPhi[i][g][2] = (2.0L/hz) * shFnDerivatives[nodeZ][dofZ][gZ] * shFnVals[nodeY][dofY][gY] * shFnVals[nodeX][dofX][gX];
                  }//end gX
                }//end gY
              }//end gZ
            }//end dofX
          }//end dofY
        }//end dofZ
      }//end nodeX
    }//end nodeY
  }//end nodeZ

  long double jac = 0.125L * hx * hy * hz;
  for(unsigned int r = 0; r < matSz; ++r) {
    for(unsigned int c = 0; c < matSz; ++c) {
      mat[r][c] = 0.0;
      for(unsigned int gZ = 0, g = 0; gZ < numGaussPts; ++gZ) {
        for(unsigned int gY = 0; gY < numGaussPts; ++gY) {
          for(unsigned int gX = 0; gX < numGaussPts; ++gX, ++g) {
            long double comp1 = (gradPhi[r][g][0] * gradPhi[c][g][0]);
            long double comp2 = (gradPhi[r][g][1] * gradPhi[c][g][1]);
            long double comp3 = (gradPhi[r][g][2] * gradPhi[c][g][2]);
            long double integrand = comp1 + comp2 + comp3;
            mat[r][c] += (gWt[gZ] * gWt[gY] * gWt[gX] * jac * integrand);
          }//end gX
        }//end gY
      }//end gZ
    }//end c
  }//end r
}

void createPoisson2DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hy, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print) {
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

  std::vector<std::vector<std::vector<long double> > > shFnDerivatives(2);
  for(unsigned int node = 0; node < 2; ++node) {
    shFnDerivatives[node].resize(K + 1);
    for(unsigned int dof = 0; dof <= K; ++dof) {
      (shFnDerivatives[node][dof]).resize(numGaussPts);
      for(unsigned int g = 0; g < numGaussPts; ++g) {
        shFnDerivatives[node][dof][g] = eval1DshFnDerivative(factorialsList, node, dof, K, coeffs, gPt[g], 1);
      }//end g
    }//end dof
  }//end node

  std::vector<std::vector<std::vector<long double> > > shFnVals(2);
  for(unsigned int node = 0; node < 2; ++node) {
    shFnVals[node].resize(K + 1);
    for(unsigned int dof = 0; dof <= K; ++dof) {
      (shFnVals[node][dof]).resize(numGaussPts);
      for(unsigned int g = 0; g < numGaussPts; ++g) {
        shFnVals[node][dof][g] = eval1DshFn(node, dof, K, coeffs, gPt[g]);
      }//end g
    }//end dof
  }//end node

  std::vector<std::vector<std::vector<long double> > > gradPhi(matSz);
  for(unsigned int nodeY = 0, i = 0; nodeY < 2; ++nodeY) {
    for(unsigned int nodeX = 0; nodeX < 2; ++nodeX) {
      for(unsigned int dofY = 0; dofY <= K; ++dofY) {
        for(unsigned int dofX = 0; dofX <= K; ++dofX, ++i) {
          gradPhi[i].resize(numGaussPts*numGaussPts);
          for(unsigned int gY = 0, g = 0; gY < numGaussPts; ++gY) {
            for(unsigned int gX = 0; gX < numGaussPts; ++gX, ++g) {
              gradPhi[i][g].resize(2);
              gradPhi[i][g][0] = (2.0L/hx) * shFnVals[nodeY][dofY][gY] * shFnDerivatives[nodeX][dofX][gX];
              gradPhi[i][g][1] = (2.0L/hy) * shFnDerivatives[nodeY][dofY][gY] * shFnVals[nodeX][dofX][gX];
            }//end gX
          }//end gY
        }//end dofX
      }//end dofY
    }//end nodeX
  }//end nodeY

  long double jac = 0.25L * hx * hy;
  for(unsigned int r = 0; r < matSz; ++r) {
    for(unsigned int c = 0; c < matSz; ++c) {
      mat[r][c] = 0.0;
      for(unsigned int gY = 0, g = 0; gY < numGaussPts; ++gY) {
        for(unsigned int gX = 0; gX < numGaussPts; ++gX, ++g) {
          long double comp1 = (gradPhi[r][g][0] * gradPhi[c][g][0]);
          long double comp2 = (gradPhi[r][g][1] * gradPhi[c][g][1]);
          long double integrand = comp1 + comp2;
          mat[r][c] += (gWt[gY] * gWt[gX] * jac * integrand);
        }//end gX
      }//end gY
    }//end c
  }//end r
}

void createPoisson1DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print) {
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

  std::vector<std::vector<std::vector<long double> > > shFnDerivatives(2);
  for(unsigned int node = 0; node < 2; ++node) {
    shFnDerivatives[node].resize(K + 1);
    for(unsigned int dof = 0; dof <= K; ++dof) {
      (shFnDerivatives[node][dof]).resize(numGaussPts);
      for(unsigned int g = 0; g < numGaussPts; ++g) {
        shFnDerivatives[node][dof][g] = eval1DshFnDerivative(factorialsList, node, dof, K, coeffs, gPt[g], 1);
      }//end g
    }//end dof
  }//end node

  std::vector<std::vector<long double> > gradPhi(matSz);
  for(unsigned int node = 0, i = 0; node < 2; ++node) {
    for(unsigned int dof = 0; dof <= K; ++dof, ++i) {
      gradPhi[i].resize(numGaussPts);
      for(unsigned int g = 0; g < numGaussPts; ++g) {
        gradPhi[i][g] = (2.0L/hx) * shFnDerivatives[node][dof][g];
      }//end g
    }//end dof
  }//end node

  long double jac = 0.5L * hx;
  for(unsigned int r = 0; r < matSz; ++r) {
    for(unsigned int c = 0; c < matSz; ++c) {
      mat[r][c] = 0.0;
      for(unsigned int g = 0; g < numGaussPts; ++g) {
        mat[r][c] += (gWt[g] * jac * gradPhi[r][g] * gradPhi[c][g]);
      }//end g
    }//end c
  }//end r
}

long double eval1DshFnDerivative(std::vector<unsigned long long int>& factorialsList, unsigned int nodeId,
    unsigned int dofId, unsigned int K, std::vector<long long int> & coeffs, long double xi, unsigned int l) {
#ifdef DEBUG
  assert(nodeId < 2);
  assert(dofId <= K);
  assert(K <= 10);
  //xi is the coordinate in the reference element.
  assert(xi >= -1.0L);
  assert(xi <= 1.0L);
  assert( (coeffs.size()) == (8*(K + 1)*(K + 1)) );
#endif
  long double result = 0.0;
  if(l == 0) {
    result = eval1DshFn(nodeId, dofId, K, coeffs, xi); 
  } else {
    unsigned int P = (2*K) + 1;
    if(l <= P) {
      long long int* coeffArr = &(coeffs[2*(P + 1)*((nodeId*(K + 1)) + dofId)]);
      for(unsigned int i = l; i <= P; ++i) {
        long double num = coeffArr[2*i];
        long double den = coeffArr[(2*i) + 1];
        long double c = num/den;
        result += (c*powDerivative(factorialsList, xi, i, l));    
      }//end i
    }
  }
  return result;
}

long double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, long double xi) {
#ifdef DEBUG
  assert(nodeId < 2);
  assert(dofId <= K);
  assert(K <= 10);
  //xi is the coordinate in the reference element.
  assert(xi >= -1.0L);
  assert(xi <= 1.0L);
  assert( (coeffs.size()) == (8*(K + 1)*(K + 1)) );
#endif
  int P = (2*K) + 1;
  long long int* coeffArr = &(coeffs[2*(P + 1)*((nodeId*(K + 1)) + dofId)]);
  long double num = coeffArr[2*P];
  long double den = coeffArr[(2*P) + 1];
  long double c = num/den;
  long double result = c;
  for(int i = (P - 1); i >= 0; --i) {
    num = coeffArr[2*i];
    den = coeffArr[(2*i) + 1];
    c = num/den;
    result = ((result*xi) + c);
  }//end i
  return result;
}

long double powDerivative(std::vector<unsigned long long int>& factorialsList, long double x,
    unsigned int i, unsigned int l) {
  long double result;
  if(l == 0) {
    result = myIntPow(x, i);
  } else if(l > i) {
    result = 0.0;
  } else if(l == i) {
    result = static_cast<long double>(factorialsList[l]);
  } else {
    int j = (i - l);
    result = (static_cast<long double>(factorialsList[i]/factorialsList[j]))*(myIntPow(x, j));
  }
  return result;
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

void read1DshapeFnCoeffs(unsigned int K, char const * prefix, std::vector<long long int> & coeffs) {
  char fname[256];
  sprintf(fname, "%s/ShFnCoeffs1D/C%uShFnCoeffs1D.txt", prefix, K);
  FILE *fp = fopen(fname, "r"); 
#ifdef DEBUG
  assert(fp != NULL);
#endif
  unsigned int numCoeffs = 4*(K + 1)*(K + 1);
  coeffs.resize(2*numCoeffs);
  for(unsigned int i = 0; i < (2*numCoeffs); ++i) {
    fscanf(fp, "%lld", &(coeffs[i]));
  }//end i 
  fclose(fp);
}

void suppressSmallValues(const unsigned int len, double* vec) {
  for(unsigned int i = 0; i < len; ++i) {
    if(softEquals(vec[i], 0.0)) {
      vec[i] = 0.0;
    }
  }//end i
}

bool softEquals(long double a, long double b) {
  long double diff = a - b;
  bool result = false;
  if((fabs(diff)) < 1.0e-12L) {
    result = true;
  } else {
    if(fabs(a) > 1.0) {
      if(fabs(diff/a) < 1.0e-12L) {
        result = true;
      }
    }
  }
  return result;
}

long double gaussWeight(int n, long double x) {
  long double polyPrime = legendrePolyPrime(n, x);
  return (2.0L/((1.0L - (x*x))*polyPrime*polyPrime));
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

int getDofsPerNode(int dim, int K) {
  int dofsPerNode = (K + 1);
  if(dim > 1) {
    dofsPerNode *= (K + 1);
  }
  if(dim > 2) {
    dofsPerNode *= (K + 1);
  }
  return dofsPerNode;
}

long double myIntPow(long double base, unsigned int exponent) {
  long double res;
  if(exponent == 0) {
    res = 1.0;
  } else if(exponent == 1) {
    res = base;
  } else if(exponent == 2) {
    res = (base*base);
  } else if(exponent == 3) {
    res = (base*base*base);
  } else if(exponent == 4) {
    long double baseSqr = (base*base);
    res = (baseSqr*baseSqr);
  } else if(exponent == 5) {
    long double baseSqr = (base*base);
    res = (baseSqr*baseSqr*base);
  } else if(exponent == 6) {
    long double baseCube = (base*base*base);
    res = (baseCube*baseCube);
  } else if(exponent == 7) {
    long double baseCube = (base*base*base);
    res = (baseCube*baseCube*base);
  } else if(exponent == 8) {
    long double baseSqr = (base*base);
    long double baseFour = (baseSqr*baseSqr);
    res = (baseFour*baseFour);
  } else if(exponent == 9) {
    long double baseSqr = (base*base);
    long double baseFour = (baseSqr*baseSqr);
    res = (baseFour*baseFour*base);
  } else if(exponent == 10) {
    long double baseSqr = (base*base);
    long double baseFour = (baseSqr*baseSqr);
    res = (baseFour*baseFour*baseSqr);
  } else {
    res = ( (myIntPow(base, 10)) * (myIntPow(base, (exponent - 10))) );
  }
  return res;
}


