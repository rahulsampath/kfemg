
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
  assert(argc == 5);
  const int dim = atoi(argv[1]); 
  assert( (dim == 1) || (dim == 3) );
  const int K = atoi(argv[2]);
  const int N = atoi(argv[3]); //Nodes per dimension
  const int numGrids = atoi(argv[4]);

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  MyMatrix myMat;

  double computeMatStartTime = MPI_Wtime();
 // computeMatrix();
  double computeMatEndTime = MPI_Wtime();

  std::cout<<"Mat create time = "<<(computeMatEndTime - computeMatStartTime)<<std::endl;

  const int numPDEs = 2;
  int maxIterations = 1000;
  if(numGrids == 1) {
    maxIterations = 1;
  }
  const int coarseSize = 8;

  double setupStart = MPI_Wtime();
  ML_set_random_seed(123456);
  ML* ml_object;
  ML_Aggregate* agg_object;
  ML_Create(&ml_object, numGrids);
  ML_Aggregate_Create(&agg_object);

  ML_Init_Amatrix(ml_object, 0, (2*N), (2*N), &myMat);
  ML_Set_Amatrix_Getrow(ml_object, 0, &myGetRow, NULL, (2*N));
  ML_Set_Amatrix_Matvec(ml_object, 0, &myMatVec);
  ML_Set_MaxIterations(ml_object, maxIterations);
  ML_Set_Tolerance(ml_object, 1.0e-12);
  ML_Set_ResidualOutputFrequency(ml_object, 1);
  ML_Set_PrintLevel(10);
  ML_Set_OutputLevel(ml_object, 10);

  agg_object->num_PDE_eqns = numPDEs;
  agg_object->nullspace_dim = 1;
  ML_Aggregate_Set_MaxCoarseSize(agg_object, coarseSize);
  ML_Aggregate_Set_CoarsenScheme_UncoupledMIS(agg_object);

  const int nlevels = ML_Gen_MGHierarchy_UsingAggregation(ml_object, 0, ML_INCREASING, agg_object);
  std::cout<<"Number of actual levels: "<<nlevels<<std::endl;

  for(int lev = 0; lev < (nlevels - 1); ++lev) {
    ML_Gen_Smoother_SymGaussSeidel(ml_object, lev, ML_BOTH, 2, 1.0);
    //ML_Gen_Smoother_Jacobi(ml_object, lev, ML_BOTH, 2, 0.8);
  }
  ML_Gen_Smoother_Amesos(ml_object, (nlevels - 1), ML_AMESOS_KLU, -1, 0.0);

  ML_Gen_Solver(ml_object, ML_MGV, 0, (nlevels-1));

  double setupEnd = MPI_Wtime();

  double* solArr = new double[2*N];
  double* rhsArr = new double[2*N];

  for(int i = 0; i < (2*N); i++) {
    solArr[i] = (static_cast<double>(rand()))/(static_cast<double>(RAND_MAX));
  }//end for i

  myMatVec(NULL, (2*N), solArr, (2*N), rhsArr);

  for(int i = 0; i < (2*N); ++i) {
    solArr[i] = 0.0;
  }//end for i

  double solveStart = MPI_Wtime();

  ML_Iterate(ml_object, solArr, rhsArr);

  double solveEnd = MPI_Wtime();

  ML_Aggregate_Destroy(&agg_object);
  ML_Destroy(&ml_object);

  std::cout<<"Setup Time = "<<(setupEnd - setupStart)<<std::endl;
  std::cout<<"Solve Time = "<<(solveEnd - solveStart)<<std::endl;

  delete [] solArr;
  delete [] rhsArr;
  MPI_Finalize();
}  



