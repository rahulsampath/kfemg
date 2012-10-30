
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

#define __PI__ 3.1415926535897932

void suppressSmallValues(const unsigned int len, double* vec) {
  for(int i = 0; i < len; ++i) {
    if(softEquals(vec[i], 0.0)) {
      vec[i] = 0.0;
    }
  }//end i
}

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

void printVector(const unsigned int len, double* vec) {
  for(int i = 0; i < len; ++i) {
    std::cout<<(std::setprecision(13))<<(vec[i])<<std::endl;
  }//end i
}

double computeLambda(const unsigned int K, const unsigned int Nx, double* inArr, double* outArr) {
  double lambda;
  bool set = false;
  for(int i = 1; i < (Nx - 1); ++i) {
    for(int d = 0; d <= K; ++d) {
      int dof = ((K + 1)*i) + d;
      if(fabs(inArr[dof]) > 0) {
        double tmp = (outArr[dof])/(inArr[dof]);
        if(set) {
          if(!(softEquals(lambda, tmp))) {
            std::cout<<"Failed for i = "<<i<<" d = "<<d<<std::endl;
            std::cout<<std::setprecision(13)<<"lambda = "<<lambda<<" tmp = "<<tmp<<std::endl;
            assert(false);
          }
        } else {
          set = true;
          lambda = tmp;
        }
      }
    }//end d
  }//end i
  assert(set);
  {
    int i = 0;
    for(int d = 0; d <= K; ++d) {
      int dof = ((K + 1)*i) + d;
      if(fabs(inArr[dof]) > 0) {
        double tmp = 0.5*lambda*inArr[dof];
        if(!(softEquals(tmp, outArr[dof]))) {
          std::cout<<"Failed for i = "<<i<<" d = "<<d<<std::endl;
          std::cout<<std::setprecision(13)<<"tmp = "<<tmp<<" outArrVal = "<<(outArr[dof])<<std::endl;
          assert(false);
        }
      }
    }//end d
  }
  {
    int i = Nx - 1;
    for(int d = 0; d <= K; ++d) {
      int dof = ((K + 1)*i) + d;
      if(fabs(inArr[dof]) > 0) {
        double tmp = 0.5*lambda*inArr[dof];
        if(!(softEquals(tmp, outArr[dof]))) {
          std::cout<<"Failed for i = "<<i<<" d = "<<d<<std::endl;
          std::cout<<std::setprecision(13)<<"tmp = "<<tmp<<" outArrVal = "<<(outArr[dof])<<std::endl;
          assert(false);
        }
      }
    }//end d
  }
  return lambda;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if(argc <= 2) {
    std::cout<<"USAGE: <exe> K Nx"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;
  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;
  const unsigned int Nx = atoi(argv[2]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;
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
  double* blkOutArr = new double[vecLen];

  for(int wNum = 0; wNum < Nx; ++wNum) {
    for(int wDof = 0; wDof <= K; ++wDof) {
      std::cout<<"Testing: ("<<wNum<<", "<<wDof<<")"<<std::endl;
      setInputVector(wNum, wDof, K, Nx, inArr);

      std::cout<<"Input Vector: "<<std::endl;
      printVector(vecLen, inArr);

      myMatVecPrivate(&myMat, vecLen, inArr, outArr);
      suppressSmallValues(vecLen, outArr); 

      std::cout<<"Output Vector: "<<std::endl;
      printVector(vecLen, outArr);

      myBlockMatVec(&myMat, dofsPerNode, wDof, vecLen, inArr, blkOutArr);
      suppressSmallValues(vecLen, blkOutArr); 

      std::cout<<"Block Output Vector: "<<std::endl;
      printVector(vecLen, blkOutArr);

      if((wDof == 0) && ((wNum == 0) || (wNum == (Nx - 1)))) {
        for(int i = 0; i < vecLen; ++i) {
          assert(softEquals(inArr[i], outArr[i]));
        }//end i
      } else {
        double lambda = computeLambda(K, Nx, inArr, outArr);
        std::cout<<"Lambda = "<<std::setprecision(13)<<lambda<<std::endl;
      }

      for(int i = 0; i < vecLen; ++i) {
        if((i%dofsPerNode) == wDof) {
          assert(softEquals(outArr[i], blkOutArr[i]));
        } else {
          assert(softEquals(blkOutArr[i], 0.0));
        }
      }//end i

      std::cout<<std::endl;
    }//end wDof
  }//end wNum

  std::cout<<"Pass!"<<std::endl;

  delete [] inArr;
  delete [] outArr;
  delete [] blkOutArr;

  MPI_Finalize();

  return 0;
}  




