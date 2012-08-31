
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
  const unsigned int numGrids = atoi(argv[3]);
  const unsigned int Nx = atoi(argv[4]); 
  unsigned int Ny = 1;
  unsigned int Nz = 1;
  if(dim == 3) {
    Ny = atoi(argv[5]);
    Nz = atoi(argv[6]);
  }
  bool useRandomRHS = atoi(argv[7]);

  double hx = 1.0/(static_cast<double>(Nx - 1));
  double hy, hz;
  if(dim == 3) {
    hy = 1.0/(static_cast<double>(Ny - 1));
    hz = 1.0/(static_cast<double>(Nz - 1));
  }

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<std::vector<double> > elemMat;
  if(dim == 1) {
    createPoisson1DelementMatrix(K, coeffs, hx, elemMat);
  } else {
    createPoisson3DelementMatrix(K, coeffs, hz, hy, hx, elemMat);
  }

  double computeMatStartTime = MPI_Wtime();
  MyMatrix myMat;
  // computeMatrix();
  double computeMatEndTime = MPI_Wtime();

  std::cout<<"Mat create time = "<<(computeMatEndTime - computeMatStartTime)<<std::endl;

  unsigned int maxIters = 1000;
  if(numGrids == 1) {
    maxIters = 1;
  }

  double setupStart = MPI_Wtime();
  ML* ml_obj;
  ML_Aggregate* agg_obj;
  createMLobjects(ml_obj, agg_obj, numGrids, maxIters, dim, K, myMat);
  double setupEnd = MPI_Wtime();

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

  ML_Iterate(ml_obj, solArr, rhsArr);

  double solveEnd = MPI_Wtime();

  destroyMLobjects(ml_obj, agg_obj);

  std::cout<<"Setup Time = "<<(setupEnd - setupStart)<<std::endl;
  std::cout<<"Solve Time = "<<(solveEnd - solveStart)<<std::endl;

  delete [] solArr;
  delete [] rhsArr;

  MPI_Finalize();
}  



