
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
  double* diag = new double[vecLen];

  getDiagonal(&myMat, vecLen, diag);

  for(int i = 0; i < vecLen; ++i) {
    for(int j = 0; j < (myMat.vals)[i].size(); ++j) {
      if(diag[i] < fabs((myMat.vals)[i][j])) {
        std::cout<<"Diag = "<<(diag[i])<<" Other = "<<((myMat.vals)[i][j])<<std::endl;
        assert(false);
      }
    }//end j
  }//end i

  std::cout<<"Pass!"<<std::endl;

  delete [] diag;

  MPI_Finalize();

  return 0;
}  



