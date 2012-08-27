
#ifndef __KFEMG_UTILS__
#define __KFEMG_UTILS__

#include <vector>

void read1DshapeFnCoeffs(int K, std::vector<long long int> & coeffs);

void gaussQuad(std::vector<double> & x, std::vector<double> & w);

#endif

