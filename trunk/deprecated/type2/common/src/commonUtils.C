
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cmath>
#include <vector>
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

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
              gradPhi[i][g][0] = (2.0L/hx) * myIntPow((0.5L * hx), dofX) * 
                myIntPow((0.5L * hy), dofY) * shFnVals[nodeY][dofY][gY] * shFnDerivatives[nodeX][dofX][gX];
              gradPhi[i][g][1] = (2.0L/hy) * myIntPow((0.5L * hx), dofX) * 
                myIntPow((0.5L * hy), dofY) * shFnDerivatives[nodeY][dofY][gY] * shFnVals[nodeX][dofX][gX];
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
        gradPhi[i][g] = (2.0L/hx) * myIntPow((0.5L * hx), dof) * shFnDerivatives[node][dof][g];
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

void suppressSmallValues(const unsigned int len, double* vec) {
  for(unsigned int i = 0; i < len; ++i) {
    if(softEquals(vec[i], 0.0)) {
      vec[i] = 0.0;
    }
  }//end i
}

void read1DshapeFnCoeffs(unsigned int K, std::vector<long long int> & coeffs) {
  char fname[256];
  sprintf(fname, "../../common/ShFnCoeffs1D/C%uShFnCoeffs1D.txt", K);

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
    result = ((static_cast<long double>(factorialsList[i]))/(static_cast<long double>(factorialsList[j])))*(myIntPow(x, j));
  }

  return result;
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
#ifdef DEBUG
  assert(dim > 0);
  assert(dim <= 3);
#endif
  int dofsPerNode = (K + 1);
  if(dim > 1) {
    dofsPerNode *= (K + 1);
  }
  if(dim > 2) {
    dofsPerNode *= (K + 1);
  }
  return dofsPerNode;
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




