
#ifndef __LOA__
#define __LOA__

struct LOAdata {
  int K;
  std::vector<std::vector<long long int> >* coeffs;
};

void setupLOA(LOAdata* data, int K, std::vector<std::vector<long long int> >& coeffs);

void destroyLOA(LOAdata* data);

#endif

