
#ifndef __COMMON_UTILS__
#define __COMMON_UTILS__

#include <vector>

double legendrePoly(int n, double x);

double legendrePolyPrime(int n, double x); 

double gaussWt(int n, double x); 

bool softEquals(double a, double b);

void read1DshapeFnCoeffs(int K, std::vector<long long int> & coeffs);

void gaussQuad(std::vector<double> & x, std::vector<double> & w);

#endif

