
#include <iostream>
#include <cstdio>
#include <vector>
#include "common/include/commonUtils.h"

int main() {
  int len = 21;
  std::cout<<std::endl;
  std::cout<<"void initFactorials(std::vector<unsigned long long int>& fac) { "<<std::endl;
  std::cout<<"//REMARK: 21! requires more than 64 bits."<<std::endl;
  std::cout<<"fac.resize("<<len<<");"<<std::endl;
  for(int i = 0; i < len; ++i) {
    unsigned long long int val = factorial(i);
    printf("fac[%d] = %lluULL;\n", i, val);
  }//end i
  std::cout<<" } "<<std::endl<<std::endl;
}

