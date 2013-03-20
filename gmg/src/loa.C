
#include "gmg/include/loa.h"

void setupLOA(LOAdata* data, int K, std::vector<std::vector<long long int> >& coeffs) {
  data->K = K;
  data->coeffs = &coeffs;
}

void destroyLOA(LOAdata* data) {
  delete data;
}



