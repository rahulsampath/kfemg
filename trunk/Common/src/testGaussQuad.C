
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>
#include "kfemgUtils.h"

bool softEquals(double a, double b) {
  return ((fabs(a - b)) < 1.0e-14);
}

double legendrePoly(int n, double x) {
  double result;
  if(n == 0) {
    result = 1.0;
  } else if(n == 1) {
    result = x;
  } else {
    int m = n - 1;
    result = ( ((static_cast<double>((2*m) + 1))*x*legendrePoly(m, x)) -
        (m*legendrePoly((m - 1), x)) )/(static_cast<double>(m + 1));
  }
  return result;
}

double legendrePolyPrime(int n, double x) {
  double result;
  if(n == 0) {
    result = 0.0;
  } else if(n == 1) {
    result = 1.0;
  } else {
    int m = n - 1;
    result = ( ((static_cast<double>((2*m) + 1))*legendrePoly(m, x)) +
        ((static_cast<double>((2*m) + 1))*x*legendrePolyPrime(m, x)) - 
        (m*legendrePolyPrime((m - 1), x)) )/(static_cast<double>(m + 1));
  }
  return result;
}

double gaussWt(int n, double x) {
  double polyPrime = legendrePolyPrime(n, x);
  return (2.0/((1.0 - (x*x))*polyPrime*polyPrime));
}

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


