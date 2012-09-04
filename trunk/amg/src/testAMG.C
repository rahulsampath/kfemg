
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstdlib>
#include "mpi.h"
#include "ml_include.h"
#include "common/include/commonUtils.h"
#include "amg/include/amgUtils.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  assert(argc > 4);
  const unsigned int dim = atoi(argv[1]); 
  assert( (dim == 1) || (dim == 3) );
  const unsigned int K = atoi(argv[2]);
  bool useRandomRHS = atoi(argv[3]);
  const unsigned int Nx = atoi(argv[4]); 
  unsigned int Ny = 1;
  unsigned int Nz = 1;
  unsigned int numGrids = 20;
  bool useMLasPC = true;
  if(dim == 3) {
    assert(argc > 6);
    Ny = atoi(argv[5]);
    Nz = atoi(argv[6]);
    if(argc > 7) {
      numGrids = atoi(argv[7]);
    }
    if(argc > 8) {
      useMLasPC = atoi(argv[8]);
    }
  } else {
    if(argc > 5) {
      numGrids = atoi(argv[5]);
    }
    if(argc > 6) {
      useMLasPC = atoi(argv[6]);
    }
  }

  double hx = 1.0/(static_cast<double>(Nx - 1));
  double hy, hz;
  if(dim == 3) {
    hy = 1.0/(static_cast<double>(Ny - 1));
    hz = 1.0/(static_cast<double>(Nz - 1));
  }

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  double createElemMatStartTime = MPI_Wtime();
  std::vector<std::vector<double> > elemMat;
  if(dim == 1) {
    createPoisson1DelementMatrix(K, coeffs, hx, elemMat);
  } else {
    createPoisson3DelementMatrix(K, coeffs, hz, hy, hx, elemMat);
  }
  double createElemMatEndTime = MPI_Wtime();

  double assemblyStartTime = MPI_Wtime();
  MyMatrix myMat;
  assembleMatrix(myMat, elemMat, K, dim, Nx, Ny, Nz);
  dirichletMatrixCorrection(myMat, K, dim, Nx, Ny, Nz);
  double assemblyEndTime = MPI_Wtime();

  unsigned int maxIters = 1000;
  if(numGrids == 1) {
    maxIters = 1;
  }

  double mlSetupStart = MPI_Wtime();
  ML* ml_obj;
  ML_Aggregate* agg_obj;
  createMLobjects(ml_obj, agg_obj, numGrids, maxIters, dim, K, myMat);
  double mlSetupEnd = MPI_Wtime();

  double krylovSetupStart = MPI_Wtime();
  ML_Krylov* krylov_obj = NULL;
  if(useMLasPC) {
    createKrylovObject(krylov_obj, ml_obj);
  }
  double krylovSetupEnd = MPI_Wtime();

  double* solArr = new double[(myMat.vals).size()];
  double* rhsArr = new double[(myMat.vals).size()];

  if(useRandomRHS) {
    computeRandomRHS(rhsArr, myMat);
  } else {
    assert(false);
  }

  for(int i = 0; i < (myMat.vals).size(); ++i) {
    solArr[i] = 0.0;
  }//end for i

  double solveStart = MPI_Wtime();
  if(useMLasPC) {
    ML_Krylov_Solve(krylov_obj, ((myMat.vals).size()), rhsArr, solArr);
  } else {
    ML_Iterate(ml_obj, solArr, rhsArr);
  }
  double solveEnd = MPI_Wtime();

  destroyMLobjects(ml_obj, agg_obj);

  if(useMLasPC) {
    ML_Krylov_Destroy(&krylov_obj);
  }

  std::cout<<"Element Matrix Computation Time = "<<(createElemMatEndTime - createElemMatStartTime)<<std::endl; 
  std::cout<<"Assembly Time = "<<(assemblyEndTime - assemblyStartTime)<<std::endl; 
  std::cout<<"ML Setup Time = "<<(mlSetupEnd - mlSetupStart)<<std::endl;
  std::cout<<"Krylov Setup Time = "<<(krylovSetupEnd - krylovSetupStart)<<std::endl;
  std::cout<<"Solve Time = "<<(solveEnd - solveStart)<<std::endl;

  delete [] solArr;
  delete [] rhsArr;

  MPI_Finalize();
}  



