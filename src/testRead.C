
#include "kfemgUtils.h"
#include <cassert>
#include <iostream>

int main() {
  std::vector<long long int> coeffs;

  read1DshapeFnCoeffs(1, coeffs);
  assert(coeffs.size() == 32);
  assert(coeffs[0] == 1LL);
  assert(coeffs[31] == 4LL);

  read1DshapeFnCoeffs(10, coeffs);
  assert(coeffs.size() == 968);
  assert(coeffs[0] == 1LL);
  assert(coeffs[967] == 7431782400LL);

  std::cout<<"Pass!"<<std::endl;
  return 1;
}

