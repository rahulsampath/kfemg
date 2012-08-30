
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <vector>
#include "common/include/commonUtils.h"

double eval3DshFnDerivative(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, double zi, double yi, double xi,
    unsigned int zl, unsigned int yl, unsigned int xl, double hz, double hy, double hx) {

  double result = ;

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

double eval1DshFnDerivative(unsigned int nodeId, unsigned int dofId, unsigned int K,
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

void read1DshapeFnCoeffs(int K, std::vector<long long int> & coeffs) {
  char fname[256];
  sprintf(fname, "../../common/ShFnCoeffs1D/C%dShFnCoeffs1D.txt", K);

  FILE *fp = fopen(fname, "r"); 

  assert(fp != NULL);

  int numCoeffs = 4*(K + 1)*(K + 1);

  coeffs.resize(2*numCoeffs);
  for(int i = 0; i < (2*numCoeffs); ++i) {
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

double gaussWt(int n, double x) {
  double polyPrime = legendrePolyPrime(n, x);
  return (2.0/((1.0 - (x*x))*polyPrime*polyPrime));
}



