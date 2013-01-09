
#ifndef __COMMON_UTILS__
#define __COMMON_UTILS__

#include <vector>

#define __PI__ 3.1415926535897932L

void createPoisson2DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hy, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print);

void createPoisson1DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print);

long double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, long double xi);

long double eval1DshFnDerivative(std::vector<unsigned long long int>& factorialsList, unsigned int nodeId,
    unsigned int dofId, unsigned int K, std::vector<long long int> & coeffs, long double xi, unsigned int l);

long double powDerivative(std::vector<unsigned long long int>& factorialsList, long double x,
    unsigned int i, unsigned int l);

void initFactorials(std::vector<unsigned long long int>& fac); 

inline unsigned long long int factorial(unsigned long long int n) {
  return ( (n <= 1) ? 1 : (n*factorial(n - 1)) );
}

inline long double coordGlobalToLocal(long double xg, long double xa, long double hx) {
  return ( ((xg - xa)*2.0L/hx) - 1.0L );
}

inline long double coordLocalToGlobal(long double xi, long double xa, long double hx) {
  return ( xa + (hx*(1.0L + xi)/2.0L) );
}

long double myIntPow(long double base, unsigned int exponent);

int getDofsPerNode(int dim, int K);

void read1DshapeFnCoeffs(unsigned int K, std::vector<long long int> & coeffs);

bool softEquals(long double a, long double b);

void suppressSmallValues(const unsigned int len, double* vec);

long double legendrePoly(int n, long double x);

long double legendrePolyPrime(int n, long double x); 

long double gaussWeight(int n, long double x); 

void gaussQuad(std::vector<long double> & x, std::vector<long double> & w);

#endif
