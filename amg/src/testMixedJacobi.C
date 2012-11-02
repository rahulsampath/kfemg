
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
    std::cout<<"USAGE: <exe> K Nx maxIters wNum"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  assert((K == 4) || (K == 6));
  std::cout<<"K = "<<K<<std::endl;

  const unsigned int Nx = atoi(argv[2]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;

  const unsigned int maxIters = atoi(argv[3]);
  std::cout<<"MaxIters = "<<maxIters<<std::endl;

  const unsigned int wNum = atoi(argv[4]);
  std::cout<<"wNum = "<<wNum<<std::endl;
  assert(wNum < Nx);

  const unsigned int wDof = K;
  std::cout<<"wDof = "<<wDof<<std::endl;

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

  setInputVector(wNum, wDof, K, Nx, inArr);

  double norm = maxNorm(vecLen, inArr);
  std::cout<<"Initial maxNorm = "<<std::setprecision(13)<<norm<<std::endl;
  double prevNorm = norm;

  double dampFac;
  if(K == 4) {
    dampFac = 0.333*0.3333*0.333;
  } else {
    dampFac = 0.3333*0.3333*0.3333*0.3333*0.3333*0.3333*0.3333;
  }

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
    if(K == 4) {
      applyBlockJacobi(1.0, &myMat, diag, dofsPerNode, wDof, vecLen, inPtr, outPtr);
      norm = maxNorm(vecLen, outPtr);
      std::cout<<"Step 1: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(3.0, &myMat, diag, dofsPerNode, wDof, vecLen, outPtr, inPtr);
      norm = maxNorm(vecLen, inPtr);
      std::cout<<"Step 2: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(0.7, &myMat, diag, dofsPerNode, wDof, vecLen, inPtr, outPtr);
      norm = maxNorm(vecLen, outPtr);
      std::cout<<"Step 3: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;
    } else {
      applyBlockJacobi(1.0, &myMat, diag, dofsPerNode, wDof, vecLen, inPtr, outPtr);
      norm = maxNorm(vecLen, outPtr);
      std::cout<<"Step 1: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(4.6666666669, &myMat, diag, dofsPerNode, wDof, vecLen, outPtr, inPtr);
      norm = maxNorm(vecLen, inPtr);
      std::cout<<"Step 2: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(0.6, &myMat, diag, dofsPerNode, wDof, vecLen, inPtr, outPtr);
      norm = maxNorm(vecLen, outPtr);
      std::cout<<"Step 3: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(1.43, &myMat, diag, dofsPerNode, wDof, vecLen, outPtr, inPtr);
      norm = maxNorm(vecLen, inPtr);
      std::cout<<"Step 4: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(2.3, &myMat, diag, dofsPerNode, wDof, vecLen, inPtr, outPtr);
      norm = maxNorm(vecLen, outPtr);
      std::cout<<"Step 5: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(3.649, &myMat, diag, dofsPerNode, wDof, vecLen, outPtr, inPtr);
      norm = maxNorm(vecLen, inPtr);
      std::cout<<"Step 6: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;

      applyBlockJacobi(0.6, &myMat, diag, dofsPerNode, wDof, vecLen, inPtr, outPtr);
      norm = maxNorm(vecLen, outPtr);
      std::cout<<"Step 7: Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;
    }

    norm = maxNorm(vecLen, outPtr);
    std::cout<<"Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<" ideal = "<<(dampFac*prevNorm)<<std::endl;

    prevNorm *= dampFac;

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




