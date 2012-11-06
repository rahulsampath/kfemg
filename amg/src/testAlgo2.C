
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
  if(argc <= 5) {
    std::cout<<"USAGE: <exe> K Nx maxIters wNum wDof"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;
  assert(K < 4);

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
  const unsigned int vecLen = (myMat.vals).size();
  /*
     for(int r = 0; r < vecLen; ++r) {
     int rDof = (r%dofsPerNode);
     for(int j = 0; j < ((myMat.nzCols)[r]).size(); ++j) {
     int col = myMat.nzCols[r][j];
     int cDof = (col%dofsPerNode);
     myMat.vals[r][j] *= myIntPow((0.5*hx), (rDof + cDof));
     }//end j
     }//end r
     */
  dirichletMatrixCorrection(myMat, K, dim, Nz, Ny, Nx);

  double* inArr = new double[vecLen];
  double* outArr = new double[vecLen];
  double* diag = new double[vecLen];
  double* tmp = new double[vecLen];

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

    for(int i = 0; i < vecLen; ++i) {
      outPtr[i] = 0.0;
    }//end i

    int step = 0;
    for(int r = 0; r <= K; ++r) {
      for(int c = 0; c <= K; ++c) {
        double fac = 1.0;
        if(c < r) {
          myBlockMatVec(&myMat, dofsPerNode, c, r, vecLen, outPtr, tmp);
        } else {
          myBlockMatVec(&myMat, dofsPerNode, c, r, vecLen, inPtr, tmp);
        }
        for(int i = 0; i < Nx; ++i) {
          outPtr[(i*dofsPerNode) + r] += (tmp[(i*dofsPerNode) + r]*fac);
        }//end i
      }//end c
      double alpha;
      if(r == 0) {
        alpha = 0.7;
      } else if(r == 1) {
        alpha = 0.9;
      } else if(r == 2) {
        alpha = 1.3;
      } else if(r == 3) {
        alpha = 0.8;
      } else {
        assert(false);
      }
      for(int i = 0; i < Nx; ++i) {
        outPtr[(i*dofsPerNode) + r] = inPtr[(i*dofsPerNode) + r] - (outPtr[(i*dofsPerNode) + r]*alpha/diag[(i*dofsPerNode) + r]) ;
      }//end i
      norm = maxNorm(vecLen, outPtr);
      std::cout<<"Step "<<step<<" Iter = "<<(iter + 1)<<" maxNorm = "<<std::setprecision(13)<<norm<<std::endl;
      ++step;
    }//end r

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
  delete [] tmp;

  MPI_Finalize();

  return 0;
}  



