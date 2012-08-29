
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

  std::vector<double> gPt(n);
  std::vector<double> gWt(n);
  gaussQuad(gPt, gWt);

  for(int i = 0; i < n; ++i) {
    assert(softEquals(legendrePoly(n, gPt[i]), 0));
    assert(softEquals(gaussWt(n, gPt[i]), gWt[i]));
  }//end i

  std::cout<<"Pass!"<<std::endl;
}


