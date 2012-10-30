
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

void setInputVector(const unsigned int wNumY, const unsigned int wNumX, 
    const unsigned int wDofY, const unsigned int wDofX, const unsigned int K,
    const unsigned int Ny, const unsigned int Nx, double* inArr) {

  const unsigned int dofsPerNode = (K + 1)*(K + 1);

  const unsigned int vecLen = (dofsPerNode*Ny*Nx);
  for(int i = 0; i < vecLen; ++i) {
    inArr[i] = 0.0;
  }//end i

  for(int j = 0; j < Ny; ++j) {
    double facY = (static_cast<double>(j*wNumY))/(static_cast<double>(Ny - 1));
    double yVal;
    if(wDofY == 0){
      if((wNumY == 0) || (wNumY == (Ny - 1))) {
        if(j == wNumY) {
          yVal = 1.0;
        } else {
          yVal = 0.0;
          continue;
        }
      } else {
        yVal = sin(facY*__PI__);
      }
    } else {
      yVal = cos(facY*__PI__);
    }
    for(int i = 0; i < Nx; ++i) {
      double facX = (static_cast<double>(i*wNumX))/(static_cast<double>(Nx - 1));
      double xVal;
      if(wDofX == 0){
        if((wNumX == 0) || (wNumX == (Nx - 1))) {
          if(i == wNumX) {
            xVal = 1.0;
          } else {
            xVal = 0.0;
          }
        } else {
          xVal = sin(facX*__PI__);
        }
      } else {
        xVal = cos(facX*__PI__);
      }
      inArr[(((j*Nx) + i)*dofsPerNode) + (wDofY*(K + 1)) + wDofX] = yVal*xVal;
    }//end i
  }//end j

  suppressSmallValues(vecLen, inArr);
}

void printVector(const unsigned int len, double* vec) {
  for(int i = 0; i < len; ++i) {
    std::cout<<(std::setprecision(13))<<(vec[i])<<std::endl;
  }//end i
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if(argc <= 3) {
    std::cout<<"USAGE: <exe> K Nx Ny"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 2; 
  std::cout<<"Dim = "<<dim<<std::endl;
  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;
  const unsigned int Nx = atoi(argv[2]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;
  const unsigned int Ny = atoi(argv[3]); 
  assert(Ny > 1);
  std::cout<<"Ny = "<<Ny<<std::endl;  
  const unsigned int Nz = 1; 

  const unsigned int dofsPerNode = (K + 1)*(K + 1);

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  long double hy = 1.0L/(static_cast<long double>(Ny - 1));

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList); 

  std::vector<std::vector<long double> > elemMat;
  createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, true);

  MyMatrix myMat;
  assembleMatrix(myMat, elemMat, K, dim, Nz, Ny, Nx);
  dirichletMatrixCorrection(myMat, K, dim, Nz, Ny, Nx);

  const unsigned int vecLen = (myMat.vals).size();
  double* inArr = new double[vecLen];
  double* outArr = new double[vecLen];
  double* blkOutArr = new double[vecLen];

  for(int wNumY = 0; wNumY < Ny; ++wNumY) {
    for(int wNumX = 0; wNumX < Nx; ++wNumX) {
      for(int wDofY = 0; wDofY <= K; ++wDofY) {
        for(int wDofX = 0; wDofX <= K; ++wDofX) {
          std::cout<<"Testing: ("<<wNumY<<","<<wDofY<<"),("<<wNumX<<","<<wDofX<<")"<<std::endl;
          const unsigned int wDof = ((K + 1)*wDofY) + wDofX;

          setInputVector(wNumY, wNumX, wDofY, wDofX, K, Ny, Nx, inArr);

          //std::cout<<"Input Vector: "<<std::endl;
          //printVector(vecLen, inArr);

          myMatVecPrivate(&myMat, vecLen, inArr, outArr);
          suppressSmallValues(vecLen, outArr); 

          //std::cout<<"Output Vector: "<<std::endl;
          //printVector(vecLen, outArr);

          myBlockMatVec(&myMat, dofsPerNode, wDof, vecLen, inArr, blkOutArr);
          suppressSmallValues(vecLen, blkOutArr); 

          //std::cout<<"Block Output Vector: "<<std::endl;
          //printVector(vecLen, blkOutArr);

          std::cout<<std::endl;
        }//end wDofX
      }//end wDofY
    }//end wNumX
  }//end wNumY

  std::cout<<"Pass!"<<std::endl;

  delete [] inArr;
  delete [] outArr;
  delete [] blkOutArr;

  MPI_Finalize();

  return 0;
}  




