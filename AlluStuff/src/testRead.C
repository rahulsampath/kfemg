
#include "kfemgUtils.h"
#include <cassert>
#include <iostream>

int main() {
  std::vector<long long int> coeffs;

  std::cout << "Before Reading "<< std::endl;
  read1DshapeFnCoeffs(1, coeffs);
  std::cout << "After Reading "<< std::endl;
  assert(coeffs.size() == 96);
  assert(coeffs[0] == 1LL);
  assert(coeffs[31] == 4LL);

  read1DshapeFnCoeffs(10, coeffs);
  assert(coeffs.size() == 968*3);
  assert(coeffs[0] == 1LL);
  assert(coeffs[967] == 7431782400LL);

  std::cout<<"Pass!"<<std::endl;
  return 1;
}

