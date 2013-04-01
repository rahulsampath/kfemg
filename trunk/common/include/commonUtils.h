
#ifndef __COMMON_UTILS__
#define __COMMON_UTILS__

#include <vector>

#define __PI__ 3.1415926535897932L

double det2x2(double mat[2][2]);

double det3x3(double mat[3][3]);

void eigenVals2x2(double mat[2][2], double val[2]);

void matMult3x3(double mat[3][3], double in[3], double out[3]);

void matMult2x2(double mat[2][2], double in[2], double out[2]);

void matInvert2x2(double mat[2][2], double matInv[2][2]);

void matInvert3x3(double mat[3][3], double matInv[3][3]);

long double legendrePolyPrime(int n, long double x); 

long double legendrePoly(int n, long double x);

long double gaussWeight(int n, long double x); 

void gaussQuad(std::vector<long double> & x, std::vector<long double> & w);

void createPoisson1DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print);

void createPoisson2DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hy, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print);

void createPoisson3DelementMatrix(std::vector<unsigned long long int>& factorialsList,
    unsigned int K, std::vector<long long int> & coeffs, long double hz, long double hy, long double hx, 
    std::vector<std::vector<long double> >& mat, bool print);

long double eval1DshFn(unsigned int nodeId, unsigned int dofId, unsigned int K, 
    std::vector<long long int> & coeffs, long double xi);

long double eval1DshFnDerivative(std::vector<unsigned long long int>& factorialsList, unsigned int nodeId,
    unsigned int dofId, unsigned int K, std::vector<long long int> & coeffs, long double xi, unsigned int l);

long double powDerivative(std::vector<unsigned long long int>& factorialsList, long double x,
    unsigned int i, unsigned int l);

int getDofsPerNode(int dim, int K);

void initFactorials(std::vector<unsigned long long int>& fac); 

long double myIntPow(long double base, unsigned int exponent);

void read1DshapeFnCoeffs(unsigned int K, char const * prefix, std::vector<long long int> & coeffs);

void suppressSmallValues(const unsigned int len, double* vec);

bool softEquals(long double a, long double b);

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



