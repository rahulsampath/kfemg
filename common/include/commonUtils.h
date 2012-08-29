
#ifndef __COMMON_UTILS__
#define __COMMON_UTILS__

#include <vector>

double legendrePoly(int n, double x);

double legendrePolyPrime(int n, double x); 

double gaussWt(int n, double x); 

bool softEquals(double a, double b);

void read1DshapeFnCoeffs(int K, std::vector<long long int> & coeffs);

void gaussQuad(std::vector<double> & x, std::vector<double> & w);

double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, double x);

double eval1DshFnDerivative(unsigned int nodeId, unsigned int dofId, unsigned int K,
    std::vector<long long int> & coeffs, double x, unsigned int l);

double powDerivative(double x, unsigned int i, unsigned int l);

inline unsigned long long int factorial(unsigned long long int n) {
  return ( (n <= 1) ? 1 : (n*factorial(n - 1)) );
}

#endif

