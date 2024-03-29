
#include <cassert>
#include <iostream>
#include "common/include/commonUtils.h"

int main() {
  std::vector<long long int> coeffs;

  std::cout<<"sizeof(long long int) = "<<(sizeof(long long int))<<std::endl;

  read1DshapeFnCoeffs(1, "../../common", coeffs);
  assert(coeffs.size() == 32);
  assert(coeffs[0] == 1LL);
  assert(coeffs[31] == 4LL);

  read1DshapeFnCoeffs(10, "../../common", coeffs);
  assert(coeffs.size() == 968);
  assert(coeffs[0] == 1LL);
  assert(coeffs[967] == 7431782400LL);

  std::cout<<"Pass!"<<std::endl;
  return 1;
}

