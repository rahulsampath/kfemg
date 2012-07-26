
#include "kfemgUtils.h"
#include <cstdio>
#include <cstring>
#include <iostream>

void read1DshapeFnCoeffs(int K, std::vector<long long int> & coeffs) {
  char fname[256];
  sprintf(fname, "C%dShFnCoeffs1D.txt", K);

  FILE *fp = fopen(fname, "r"); 

  int numCoeffs = 4*(K + 1)*(K + 1)*(K + 2);

  coeffs.resize(2*numCoeffs);
  for(int i = 0; i < (2*numCoeffs); ++i) {
    std::cout << "Read Coeff" << i << std::endl;
    fscanf(fp, "%lld", &(coeffs[i]));
  }//end i 

  fclose(fp);
}


