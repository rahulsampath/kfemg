
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>
#include "common/include/commonUtils.h"

int main(int argc, char**argv) {
  assert(argc == 2);
  int n = atoi(argv[1]);

  std::cout<<"Testing "<<n<<" Point Rule."<<std::endl;
  std::cout<<"Sizeof(double) = "<<(sizeof(double))<<std::endl;
  std::cout<<"Sizeof(long double) = "<<(sizeof(long double))<<std::endl;

  std::vector<long double> gPt(n);
  std::vector<long double> gWt(n);
  gaussQuad(gPt, gWt);

  for(int i = 0; i < n; ++i) {
    assert(softEquals(legendrePoly(n, gPt[i]), 0));
    assert(softEquals(gaussWeight(n, gPt[i]), gWt[i]));
  }//end i

  std::cout<<"Pass!"<<std::endl;
}


