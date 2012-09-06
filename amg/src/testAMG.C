
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
    std::cout<<"USAGE: <exe> dim K useRandomRHS Nx (Ny) (Nz) [numGrids] [useMLasPC] [maxIters] [rTol]."<<std::endl;
    std::cout<<"[]: Optional. (): Depends on dim."<<std::endl;
    assert(false);
  }
  const unsigned int dim = atoi(argv[1]); 
  assert(dim > 0);
  assert( (dim == 1) || (dim == 3) );
  std::cout<<"Dim = "<<dim<<std::endl;
  const unsigned int K = atoi(argv[2]);
  std::cout<<"K = "<<K<<std::endl;
  bool useRandomRHS = atoi(argv[3]);
  std::cout<<"Random-RHS = "<<useRandomRHS<<std::endl;
  const unsigned int Nx = atoi(argv[4]); 
  assert(Nx > 1);
  std::cout<<"Nx = "<<Nx<<std::endl;
  unsigned int Ny = 1;
  if(dim > 1) {
    assert(argc > 5);
    Ny = atoi(argv[5]);
    assert(Ny > 1);
  }
  std::cout<<"Ny = "<<Ny<<std::endl;
  unsigned int Nz = 1;
  if(dim > 2) {
    assert(argc > 6);
    Nz = atoi(argv[6]);
    assert(Nz > 1);
  }
  std::cout<<"Nz = "<<Nz<<std::endl;
  unsigned int numGrids = 20;
  if(argc > (4 + dim)) {
    numGrids = atoi(argv[(4 + dim)]);
  }
  std::cout<<"Max-Num-Grids = "<<numGrids<<std::endl;
  bool useMLasPC = true;
  if(argc > (5 + dim)) {
    useMLasPC = atoi(argv[(5 + dim)]);
  }
  std::cout<<"ML-as-PC = "<<useMLasPC<<std::endl;
  unsigned int maxIters = 10000;
  if(argc > (6 + dim)) {
    maxIters = atoi(argv[(6 + dim)]);
  }
  if(numGrids == 1) {
    maxIters = 1;
  }
  std::cout<<"MaxIters = "<<maxIters<<std::endl;
  double rTol = 1.0e-6;
  if(argc > (7 + dim)) {
    rTol = atof(argv[(7 + dim)]);
  }
  std::cout<<"R-Tol = "<<rTol<<std::endl;

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

#ifdef __PRINT_MAT__
  std::cout<<"Element Matrix: "<<std::endl;
  for(int i = 0; i < elemMat.size(); ++i) {
    for(int j = 0; j < elemMat[i].size(); ++j) {
      std::cout<<"E["<<i<<"]["<<j<<"] = "<<std::setprecision(15)<<(elemMat[i][j])<<std::endl;
    }//end j
    std::cout<<std::endl;
  }//end i
#endif

  double assemblyStartTime = MPI_Wtime();
  MyMatrix myMat;
  assembleMatrix(myMat, elemMat, K, dim, Nx, Ny, Nz);
  double assemblyEndTime = MPI_Wtime();
  double applyBCstartTime = MPI_Wtime();
  dirichletMatrixCorrection(myMat, K, dim, Nx, Ny, Nz);
  double applyBCendTime = MPI_Wtime();
#ifdef __PRINT_MAT__
  std::cout<<"Assembled Matrix: "<<std::endl;
  printMatrix(myMat);
#endif

  double mlSetupStart = MPI_Wtime();
  ML* ml_obj;
  ML_Aggregate* agg_obj;
  createMLobjects(ml_obj, agg_obj, numGrids, maxIters, rTol, dim, K, myMat);
  double mlSetupEnd = MPI_Wtime();

  double krylovSetupStart = MPI_Wtime();
  ML_Krylov* krylov_obj = NULL;
  if(useMLasPC) {
    createKrylovObject(krylov_obj, ml_obj, maxIters, rTol);
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



