

#ifndef __tpSHAPE_FUNCTIONS__
#define __tpSHAPE_FUNCTIONS__

#include <vector>

void tensorProduct3D( int K, const std::vector<double> & si,const std::vector<double> &eta,
    const std::vector<double> &zeta, std::vector<double> & sietazeta, int dsi, int deta, int dzeta);

#endif

