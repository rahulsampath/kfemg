
#include <iostream>
#include <vector>
#include "common/include/commonUtils.h"

int main() {
  int len = 11;
  std::cout<<std::endl;
  std::cout<<"void initFactorials(std::vector<unsigned long long int>& fac) { "<<std::endl;
  std::cout<<"fac.resize("<<len<<");"<<std::endl;
  for(int i = 0; i < len; ++i) {
    unsigned long long int val = factorial(i);
    std::cout<<"fac["<<i<<"] = "<<val<<"ULL;"<<std::endl;
  }//end i
  std::cout<<" } "<<std::endl<<std::endl;
}

