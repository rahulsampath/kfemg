
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

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if(argc <= 3) {
    std::cout<<"USAGE: <exe> K Nx chkBlocked"<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;

  const unsigned int Nx = atoi(argv[2]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;

  bool chkBlocked = atoi(argv[3]);
  std::cout<<"chkBlocked = "<<chkBlocked<<std::endl;

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

  double* diag = new double[vecLen];

  getDiagonal(&myMat, vecLen, diag);

  if(chkBlocked) {
    bool strict = false;
    for(int i = 0; i < vecLen; ++i) {
      double sum = 0.0;
      for(int j = 0; j < ((myMat.nzCols)[i]).size(); ++j) {
        unsigned int col = (myMat.nzCols)[i][j];
        if(col != i) {
          if((i%dofsPerNode) == (col%dofsPerNode)) {
            sum += fabs((myMat.vals)[i][j]);
          }
        }
      }//end j
      if(diag[i] < sum) {
        std::cout<<"Diag = "<<(diag[i])<<" sum = "<<sum<<std::endl;
        assert(false);
      }
      if(diag[i] > sum) {
        strict = true;
      }
    }//end i
    assert(strict);
  } else {
    for(int i = 0; i < vecLen; ++i) {
      double sum = 0.0;
      for(int j = 0; j < ((myMat.nzCols)[i]).size(); ++j) {
        unsigned int col = (myMat.nzCols)[i][j];
        if((col%dofsPerNode) > (i%dofsPerNode)) {
          sum += fabs((myMat.vals)[i][j]);
        }
        if(diag[i] < fabs((myMat.vals)[i][j])) {
          std::cout<<"Failing for i = "<<i<<" iDof = "<<(i%dofsPerNode)
            <<" col = "<<col<<" cDof = "<<(col%dofsPerNode)<<std::endl;
          std::cout<<"Diag = "<<(diag[i])<<" oth = "<<((myMat.vals)[i][j])<<std::endl;
          assert((i%dofsPerNode) > (col%dofsPerNode));
        }
      }//end j
      if(diag[i] < sum) {
        std::cout<<"Failing for i = "<<i<<" dof = "<<(i%dofsPerNode)<<std::endl;
        std::cout<<"Diag = "<<(diag[i])<<" sum = "<<sum<<std::endl;
        assert(false);
      }
    }//end i
  }

  std::cout<<"Pass!"<<std::endl;

  delete [] diag;

  MPI_Finalize();

  return 0;
}  



