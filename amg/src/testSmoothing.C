
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

void setSinInputVector(const unsigned int waveNum, const unsigned int waveDof,
    const unsigned int K, const unsigned int Nx, double* inArr) {
  for(int i = 0; i < ((K + 1)*Nx); ++i) {
    inArr[i] = 0.0;
  }//end i

  if(waveNum == 0) {
    inArr[waveDof] = 1.0;
  } else if(waveNum == (Nx - 1)) {
    inArr[((K + 1)*(Nx - 1)) + waveDof] = 1.0;
  } else {
    for(int i = 0; i < Nx; ++i) {
      double fac = (static_cast<double>(i*waveNum))/(static_cast<double>(Nx - 1));
      inArr[((K + 1)*i) + waveDof] = sin(fac*__PI__);
    }//end i
    suppressSmallValues(((K + 1)*Nx), inArr);
  }
}

void setCosInputVector(const unsigned int waveNum, const unsigned int waveDof,
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

void printVector(int Nx, int K, double* vec) {
  for(int i = 0, cnt = 0; i < Nx; ++i) {
    for(int d = 0; d <= K; ++d, ++cnt) {
      std::cout<<" i = "<<i<<" d = "<<d<<" : "<<(std::setprecision(13))<<(vec[cnt])<<std::endl;
    }//end d
  }//end i
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if(argc <= 6) {
    std::cout<<"USAGE: <exe> K Nx wDof alpha useSin print"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;

  const unsigned int Nx = atoi(argv[2]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;

  const unsigned int wDof = atoi(argv[3]); 
  assert(wDof <= K);
  std::cout<<"wDof = "<<wDof<<std::endl;

  const double alpha = atof(argv[4]);
  std::cout<<"alpha = "<<alpha<<std::endl;

  bool useSin = atoi(argv[5]);
  std::cout<<"useSin = "<<useSin<<std::endl;

  bool print = atoi(argv[6]);
  std::cout<<"print = "<<print<<std::endl;

  const unsigned int Ny = 1; 
  const unsigned int Nz = 1; 

  const unsigned int dofsPerNode = K + 1;

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList); 

  std::vector<std::vector<long double> > elemMat;
  createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, true);

  MyMatrix myMat;
  assembleMatrix(myMat, elemMat, K, dim, Nz, Ny, Nx);
  dirichletMatrixCorrection(myMat, K, dim, Nz, Ny, Nx);

  const unsigned int vecLen = (myMat.vals).size();
  double* inArr = new double[vecLen];
  double* outArr = new double[vecLen];
  double* diag = new double[vecLen];

  getDiagonal(&myMat, vecLen, diag);

  for(int wNum = 0; wNum < Nx; ++wNum) {
    if(useSin) {
      setSinInputVector(wNum, wDof, K, Nx, inArr);
    } else {
      setCosInputVector(wNum, wDof, K, Nx, inArr);
    }
    applyBlockJacobi(alpha, &myMat, diag, dofsPerNode, wDof, vecLen, inArr, outArr);
    if(print) {
      printVector(Nx, K, outArr);
    }
    double norm = maxNorm(vecLen, outArr);
    std::cout<<"wNum = "<<wNum<<" : factor = "<<std::setprecision(13)<<norm<<std::endl;
  }//end wNum

  delete [] inArr;
  delete [] outArr;
  delete [] diag;

  MPI_Finalize();

  return 0;
}  




