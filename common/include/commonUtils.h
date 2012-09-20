
#ifndef __COMMON_UTILS__
#define __COMMON_UTILS__

#include <vector>

long long int myIntPow(long long int base, unsigned int exp);

int getDofsPerNode(int dim, int K);

long double legendrePoly(int n, long double x);

long double legendrePolyPrime(int n, long double x); 

long double gaussWeight(int n, long double x); 

void gaussQuad(std::vector<long double> & x, std::vector<long double> & w);

bool softEquals(long double a, long double b);

void read1DshapeFnCoeffs(unsigned int K, std::vector<long long int> & coeffs);

void createPoisson1DelementMatrix(unsigned int K, std::vector<long long int> & coeffs,
    long double hx, std::vector<std::vector<long double> >& mat, bool print);

void createPoisson2DelementMatrix(unsigned int K, std::vector<long long int> & coeffs, 
    long double hy, long double hx, std::vector<std::vector<long double> >& mat, bool print);

void createPoisson3DelementMatrix(unsigned int K, std::vector<long long int> & coeffs, 
    long double hz, long double hy, long double hx, std::vector<std::vector<long double> >& mat, bool print);

long double eval3DshFnGderivative(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, long double zi, long double yi, long double xi,
    int zl, int yl, int xl, long double hz, long double hy, long double hx);

long double eval2DshFnGderivative(unsigned int yNodeId, unsigned int xNodeId, unsigned int yDofId, 
    unsigned int xDofId, unsigned int K, std::vector<long long int> & coeffs, long double yi, long double xi,
    int yl, int xl, long double hy, long double hx);

long double eval1DshFnGderivative(unsigned int xNodeId, unsigned int xDofId, unsigned int K, 
    std::vector<long long int> & coeffs, long double xi, int xl, long double hx);

long double eval3DshFn(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, long double zi, long double yi, long double xi);

long double eval2DshFn(unsigned int yNodeId, unsigned int xNodeId, unsigned int yDofId, unsigned int xDofId, 
    unsigned int K, std::vector<long long int> & coeffs, long double yi, long double xi);

long double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, long double xi);

long double eval1DshFnLderivative(unsigned int nodeId, unsigned int dofId, unsigned int K,
    std::vector<long long int> & coeffs, long double xi, unsigned int l);

long double powDerivative(long double x, unsigned int i, unsigned int l);

inline unsigned long long int factorial(unsigned long long int n) {
  return ( (n <= 1) ? 1 : (n*factorial(n - 1)) );
}

inline long double coordGlobalToLocal(long double xg, long double xa, long double hx) {
  return ( ((xg - xa)*2.0L/hx) - 1.0L );
}

inline long double coordLocalToGlobal(long double xi, long double xa, long double hx) {
  return ( xa + (hx*(1.0L + xi)/2.0L) );
}

#endif

