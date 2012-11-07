
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include "mpi.h"
#include "ml_include.h"
#include "common/include/commonUtils.h"
#include "amg/include/amgUtils.h"

void setInputVector(const unsigned int waveNum, const unsigned int waveDof,
    const unsigned int K, const unsigned int Nx, double* inArr) {
  for(int i = 0; i < ((K + 1)*Nx); ++i) {
    inArr[i] = 0.0;
  }//end i

  if((waveDof == 0) && (waveNum == 0)) {
    inArr[0] = 1.0;
  } else if((waveDof == 0) && (waveNum == (Nx - 1))) {
    inArr[((K + 1)*(Nx - 1))] = 1.0;
  } else {
    for(int i = 0; i < Nx; ++i) {
      double fac = (static_cast<double>(i*waveNum))/(static_cast<double>(Nx - 1));
      if(waveDof == 0) {
        inArr[((K + 1)*i)] = sin(fac*__PI__);
      } else {
        inArr[((K + 1)*i) + waveDof] = cos(fac*__PI__);
      }
    }//end i
    suppressSmallValues(((K + 1)*Nx), inArr);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if(argc <= 4) {
    std::cout<<"USAGE: <exe> K Nxf wNum wDof"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;
  assert(K <= 7);

  const unsigned int Nxf = atoi(argv[2]); 
  assert(Nxf > 1);
  std::cout<<"Nxf = "<<Nxf<<std::endl;
  const unsigned int Nxc = 1 + ((Nxf - 1)/2);
  std::cout<<"Nxc = "<<Nxc<<std::endl;

  const unsigned int wNum = atoi(argv[3]);
  std::cout<<"wNum = "<<wNum<<std::endl;
  assert(wNum < Nxf);

  const unsigned int wDof = atoi(argv[4]);
  std::cout<<"wDof = "<<wDof<<std::endl;
  assert(wDof <= K);

  const unsigned int dofsPerNode = K + 1;
  unsigned int numGaussPts = (2*K) + 2;

  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  long double hxf = 1.0L/(static_cast<long double>(Nxf - 1));
  long double hxc = 2.0*hxf;

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList);

  double* vec = new double[Nxf*dofsPerNode];
  setInputVector(wNum, wDof, K, Nxf, vec);

  for(int j = 0; j < Nxc; ++j) {
    for(int dj = 0; dj <= K; ++dj) {
      if(dj == 0) {
        if((j == 0) || (j == (Nxc - 1))) {
          continue;
        }
      }
      double res = 0.0;
      if(j > 0) {
        int cNd = 1;
        //Elem 0
        {
        }
        //Elem 1
        {
        }
      }
      if(j < (Nxc - 1)) {
        int cNd = 0;
        //Elem 2
        {
        }
        //Elem 3
        {
        }
      }
    }//end dj
  }//end j

  delete [] vec;

  MPI_Finalize();

  return 0;
}  




