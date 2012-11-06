
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
  if(argc <= 3) {
    std::cout<<"USAGE: <exe> K Nx maxAlpha"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;

  const unsigned int Nx = atoi(argv[2]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;

  const double maxAlpha = atof(argv[3]);
  std::cout<<"maxAlpha = "<<maxAlpha<<std::endl;

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

  //getDiagonal(&myMat, vecLen, diag);
  getMaxAbsRow(&myMat, vecLen, diag);

  for(double alpha = 0.1; alpha <= maxAlpha; alpha += 0.1) {
    std::cout<<"alpha = "<<alpha<<std::endl;
    for(int wDof = 0; wDof <= K; ++wDof) {
      std::cout<<"wDof = "<<wDof<<std::endl;
      for(int wNum = 0; wNum < Nx; ++wNum) {
        setInputVector(wNum, wDof, K, Nx, inArr);
        // applyJacobi(alpha, &myMat, diag, vecLen, inArr, outArr);
        applyBlockJacobi(alpha, &myMat, diag, dofsPerNode, wDof, vecLen, inArr, outArr);
        double norm = maxNorm(vecLen, outArr);
        if( (norm >= 0.99) || (norm <= 0.4) ) {
          std::cout<<"wNum = "<<wNum<<" : factor = "<<std::setprecision(13)<<norm<<std::endl;
        }
      }//end wNum
      std::cout<<std::endl;
    }//end wDof
    std::cout<<std::endl;
  }//end alpha

  delete [] inArr;
  delete [] outArr;
  delete [] diag;

  MPI_Finalize();

  return 0;
}  




