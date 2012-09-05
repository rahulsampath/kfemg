
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "ml_include.h"
#include "common/include/commonUtils.h"
#include "amg/include/amgUtils.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if(argc <= 4) {
    std::cout<<"USAGE: <exe> dim K useRandomRHS Nx (Ny) (Nz) [numGrids] [useMLasPC] [maxIters]."<<std::endl;
    std::cout<<"[]: Optional. (): Depends on dim."<<std::endl;
    assert(false);
  }
  const unsigned int dim = atoi(argv[1]); 
  assert(dim > 0);
  assert( (dim == 1) || (dim == 3) );
  const unsigned int K = atoi(argv[2]);
  bool useRandomRHS = atoi(argv[3]);
  const unsigned int Nx = atoi(argv[4]); 
  assert(Nx > 1);
  unsigned int Ny = 1;
  if(dim > 1) {
    assert(argc > 5);
    Ny = atoi(argv[5]);
    assert(Ny > 1);
  }
  unsigned int Nz = 1;
  if(dim > 2) {
    assert(argc > 6);
    Nz = atoi(argv[6]);
    assert(Nz > 1);
  }
  unsigned int numGrids = 20;
  if(argc > (4 + dim)) {
    numGrids = atoi(argv[(4 + dim)]);
  }
  bool useMLasPC = true;
  if(argc > (5 + dim)) {
    useMLasPC = atoi(argv[(5 + dim)]);
  }
  unsigned int maxIters = 10000;
  if(argc > (6 + dim)) {
    maxIters = atoi(argv[(6 + dim)]);
  }
  if(numGrids == 1) {
    maxIters = 1;
  }

  double hx = 1.0/(static_cast<double>(Nx - 1));
  double hy, hz;
  if(dim > 1) {
    hy = 1.0/(static_cast<double>(Ny - 1));
  }
  if(dim > 2) {
    hz = 1.0/(static_cast<double>(Nz - 1));
  }

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  double createElemMatStartTime = MPI_Wtime();
  std::vector<std::vector<double> > elemMat;
  if(dim == 1) {
    createPoisson1DelementMatrix(K, coeffs, hx, elemMat);
  } else if(dim == 3) {
    createPoisson3DelementMatrix(K, coeffs, hz, hy, hx, elemMat);
  } else {
    assert(false);
  }
  double createElemMatEndTime = MPI_Wtime();

  for(int i = 0; i < elemMat.size(); ++i) {
    for(int j = 0; j < elemMat[i].size(); ++j) {
      std::cout<<"E["<<i<<"]["<<j<<"] = "<<std::setprecision(15)<<(elemMat[i][j])<<std::endl;
    }//end j
    std::cout<<std::endl;
  }//end i

  double assemblyStartTime = MPI_Wtime();
  MyMatrix myMat;
  assembleMatrix(myMat, elemMat, K, dim, Nx, Ny, Nz);
  double assemblyEndTime = MPI_Wtime();
  double applyBCstartTime = MPI_Wtime();
  dirichletMatrixCorrection(myMat, K, dim, Nx, Ny, Nz);
  double applyBCendTime = MPI_Wtime();
  printMatrix(myMat);

  double mlSetupStart = MPI_Wtime();
  ML* ml_obj;
  ML_Aggregate* agg_obj;
  createMLobjects(ml_obj, agg_obj, numGrids, maxIters, dim, K, myMat);
  double mlSetupEnd = MPI_Wtime();

  double krylovSetupStart = MPI_Wtime();
  ML_Krylov* krylov_obj = NULL;
  if(useMLasPC) {
    createKrylovObject(krylov_obj, ml_obj, maxIters);
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
  std::cout<<"ApplyBC Time = "<<(applyBCendTime - applyBCstartTime)<<std::endl; 
  std::cout<<"ML Setup Time = "<<(mlSetupEnd - mlSetupStart)<<std::endl;
  std::cout<<"Krylov Setup Time = "<<(krylovSetupEnd - krylovSetupStart)<<std::endl;
  std::cout<<"Solve Time = "<<(solveEnd - solveStart)<<std::endl;

  delete [] solArr;
  delete [] rhsArr;

  MPI_Finalize();
}  



