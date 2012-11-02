
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
  if(argc <= 6) {
    std::cout<<"USAGE: <exe> K Nx maxIters wNum wDof alpha"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;
  assert(K <= 7);

  const unsigned int Nx = atoi(argv[2]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;

  const unsigned int maxIters = atoi(argv[3]);
  std::cout<<"MaxIters = "<<maxIters<<std::endl;

  const unsigned int wNum = atoi(argv[4]);
  std::cout<<"wNum = "<<wNum<<std::endl;
  assert(wNum < Nx);

  const unsigned int wDof = atoi(argv[5]);
  std::cout<<"wDof = "<<wDof<<std::endl;
  assert(wDof <= K);

  const double alpha = atof(argv[6]);
  std::cout<<"alpha = "<<alpha<<std::endl;

  const unsigned int Ny = 1; 
  const unsigned int Nz = 1; 

  const unsigned int dofsPerNode = K + 1;

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));

  std::vector<double> factors(K + 1);
  factors[0] = -0.7;
  factors[1] = -0.9;
  factors[2] = -alpha;
  //double factors[] = {, , -1.3, -0.8, -1.17647055, -0.7, -1.12, -0.7};

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

  setInputVector(wNum, wDof, K, Nx, inArr);

  double norm = maxNorm(vecLen, inArr);
  std::cout<<"Initial maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

  int iter = 0;
  for(; iter < maxIters; ++iter) {
    double* inPtr;
    double* outPtr;
    if((iter%2) == 0) {
      inPtr = inArr;
      outPtr = outArr;
    } else {
      inPtr = outArr;
      outPtr = inArr;
    }
    myMatVecPrivate(&myMat, vecLen, inPtr, outPtr);
    divideVecPointwise(vecLen, outPtr, diag); 
    for(int i = 0; i < Nx; ++i) {
      for(int d = 0; d <= K; ++d) {
        outPtr[(i*dofsPerNode) + d] *= factors[d];
      }//end d
    }//end i
    addVec(vecLen, outPtr, inPtr);
    norm = maxNorm(vecLen, outPtr);
    std::cout<<"Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;
    if(norm < 1.0e-12) {
      break;
    }
  }//end iter
  std::cout<<"Total Num Iters = "<<iter<<std::endl;

  delete [] inArr;
  delete [] outArr;
  delete [] diag;

  MPI_Finalize();

  return 0;
}  



