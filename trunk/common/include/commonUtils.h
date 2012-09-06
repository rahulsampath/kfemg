
#ifndef __COMMON_UTILS__
#define __COMMON_UTILS__

#include <vector>

double legendrePoly(int n, double x);

double legendrePolyPrime(int n, double x); 

double gaussWeight(int n, double x); 

void gaussQuad(std::vector<double> & x, std::vector<double> & w);

bool softEquals(double a, double b);

void read1DshapeFnCoeffs(unsigned int K, std::vector<long long int> & coeffs);

void createPoisson1DelementMatrix(unsigned int K, std::vector<long long int> & coeffs,
    double hx, std::vector<std::vector<double> >& mat);

void createPoisson2DelementMatrix(unsigned int K, std::vector<long long int> & coeffs, 
    double hy, double hx, std::vector<std::vector<double> >& mat);

void createPoisson3DelementMatrix(unsigned int K, std::vector<long long int> & coeffs, 
    double hz, double hy, double hx, std::vector<std::vector<double> >& mat);

double eval3DshFnGderivative(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, double zi, double yi, double xi,
    unsigned int zl, unsigned int yl, unsigned int xl, double hz, double hy, double hx);

double eval2DshFnGderivative(unsigned int yNodeId, unsigned int xNodeId, unsigned int yDofId, 
    unsigned int xDofId, unsigned int K, std::vector<long long int> & coeffs, double yi, double xi,
    unsigned int yl, unsigned int xl, double hy, double hx);

double eval1DshFnGderivative(unsigned int xNodeId, unsigned int xDofId, unsigned int K, 
    std::vector<long long int> & coeffs, double xi, unsigned int xl, double hx);

double eval3DshFn(unsigned int zNodeId, unsigned int yNodeId, unsigned int xNodeId,
    unsigned int zDofId, unsigned int yDofId, unsigned int xDofId, unsigned int K,
    std::vector<long long int> & coeffs, double zi, double yi, double xi);

double eval2DshFn(unsigned int yNodeId, unsigned int xNodeId, unsigned int yDofId, unsigned int xDofId, 
    unsigned int K, std::vector<long long int> & coeffs, double yi, double xi);

double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, double xi);

double eval1DshFnLderivative(unsigned int nodeId, unsigned int dofId, unsigned int K,
    std::vector<long long int> & coeffs, double xi, unsigned int l);

double powDerivative(double x, unsigned int i, unsigned int l);

inline unsigned long long int factorial(unsigned long long int n) {
  return ( (n <= 1) ? 1 : (n*factorial(n - 1)) );
}

inline double coordGlobalToLocal(double xg, double xa, double hx) {
  return ( ((xg - xa)*2.0/hx) - 1.0 );
}

inline double coordLocalToGlobal(double xi, double xa, double hx) {
  return ( xa + (hx*(1 + xi)/2.0) );
}

#endif

