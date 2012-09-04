
#include <cassert>
#include <iostream>
#include "ml_include.h"
#include "amg/include/amgUtils.h"

void assembleMatrix(MyMatrix & myMat, std::vector<std::vector<double> > const & elemMat, const unsigned int K, 
    const unsigned int dim, const unsigned int Nx, const unsigned int Ny, const unsigned int Nz) {
}

void dirichletMatrixCorrection(MyMatrix & myMat, const unsigned int K, const unsigned int dim,
    const unsigned int Nx, const unsigned int Ny, const unsigned int Nz) {
}

void createKrylovObject(ML_Krylov*& krylov_obj, ML* ml_obj) {
  krylov_obj = ML_Krylov_Create(ml_obj->comm);
  ML_Krylov_Set_PrintFreq(krylov_obj, 1);
  ML_Krylov_Set_Method(krylov_obj, ML_CG);
  ML_Krylov_Set_Amatrix(krylov_obj, &((ml_obj->Amat)[0]));
  ML_Krylov_Set_PreconFunc(krylov_obj, ML_MGVSolve_Wrapper);
  ML_Krylov_Set_Precon(krylov_obj, ml_obj);
  ML_Krylov_Set_Tolerance(krylov_obj, 1.0e-12);
}

void createMLobjects(ML*& ml_obj, ML_Aggregate*& agg_obj, const unsigned int numGrids, 
    const unsigned int maxIters, const unsigned int dim, const unsigned int K, MyMatrix& myMat) {
  ML_set_random_seed(123456);

  ML_Create(&ml_obj, numGrids);
  ML_Init_Amatrix(ml_obj, 0, ((myMat.vals).size()), ((myMat.vals).size()), &myMat);
  ML_Set_Amatrix_Getrow(ml_obj, 0, &myGetRow, NULL, ((myMat.vals).size()));
  ML_Set_Amatrix_Matvec(ml_obj, 0, &myMatVec);
  ML_Set_MaxIterations(ml_obj, maxIters);
  ML_Set_Tolerance(ml_obj, 1.0e-12);
  ML_Set_ResidualOutputFrequency(ml_obj, 1);
  ML_Set_PrintLevel(10);
  ML_Set_OutputLevel(ml_obj, 10);

  unsigned int numPDEs;
  unsigned int coarseSize;
  if(dim == 1) {
    numPDEs = (K + 1); //DOFs per node
    coarseSize = 3*numPDEs; //2 Elements per dim
  } else {
    numPDEs = (K + 1)*(K + 1)*(K + 1); //DOFs per node
    coarseSize = 27*numPDEs; //2 Elements per dim
  }

  ML_Aggregate_Create(&agg_obj);
  agg_obj->num_PDE_eqns = numPDEs;
  agg_obj->nullspace_dim = 1; //CHECK THIS!
  ML_Aggregate_Set_MaxCoarseSize(agg_obj, coarseSize);
  ML_Aggregate_Set_CoarsenScheme_UncoupledMIS(agg_obj);

  const unsigned int nlevels = ML_Gen_MGHierarchy_UsingAggregation(ml_obj, 0, ML_INCREASING, agg_obj);
  std::cout<<"Number of actual MG levels: "<<nlevels<<std::endl;

  for(int lev = 0; lev < (nlevels - 1); ++lev) {
    ML_Gen_Smoother_SymGaussSeidel(ml_obj, lev, ML_BOTH, 2, 1.0);
    //ML_Gen_Smoother_Jacobi(ml_obj, lev, ML_BOTH, 2, 0.8);
  }
  ML_Gen_Smoother_Amesos(ml_obj, (nlevels - 1), ML_AMESOS_KLU, -1, 0.0);

  ML_Gen_Solver(ml_obj, ML_MGV, 0, (nlevels-1));
}

void destroyMLobjects(ML*& ml_obj, ML_Aggregate*& agg_obj) {
  ML_Aggregate_Destroy(&agg_obj);
  ML_Destroy(&ml_obj);
}

void computeRandomRHS(double* rhsArr, MyMatrix & myMat) {
  const unsigned int len = (myMat.vals).size();
  double* tmpSol = new double[len];
  for(unsigned int i = 0; i < len; ++i) {
    tmpSol[i] = (static_cast<double>(rand()))/(static_cast<double>(RAND_MAX));
  }//end for i
  myMatVecPrivate(&myMat, len, tmpSol, rhsArr);
  delete [] tmpSol;
}

int myGetRow(ML_Operator* data, int N_requested_rows, int requested_rows[],
    int allocated_space, int columns[], double values[], int row_lengths[]) {
  MyMatrix* myMat = reinterpret_cast<MyMatrix*>(ML_Get_MyGetrowData(data));
  int spaceRequired = 0;
  int cnt = 0;
  for(int i = 0; i < N_requested_rows; ++i) {
    int row = requested_rows[i];
    spaceRequired += ((myMat->nzCols)[row]).size();
    if(allocated_space >= spaceRequired) {
      for(size_t j = 0; j < ((myMat->nzCols)[row]).size(); ++j) {
        columns[cnt] = (myMat->nzCols)[row][j];
        values[cnt] = (myMat->vals)[row][j];
        ++cnt;
      }//end for j
      row_lengths[i] = ((myMat->nzCols)[row]).size();
    } else {
      return 0;
    }
  }//end for i
  return 1;
}

int myMatVec(ML_Operator* data, int in_length, double in[], int out_length, double out[]) {
  MyMatrix* myMat = reinterpret_cast<MyMatrix*>(ML_Get_MyMatvecData(data));
  myMatVecPrivate(myMat, out_length, in, out); 
  return 0;
}

void myMatVecPrivate(MyMatrix* myMat, const unsigned int len, double* in, double* out) {
  for(int i = 0; i < len; ++i) {
    out[i] = 0.0;
    for(size_t j = 0; j < ((myMat->nzCols)[i]).size(); ++j) {
      out[i] += ( ((myMat->vals)[i][j]) * (in[(myMat->nzCols)[i][j]]) );
    }//end for j
  }//end for i
}


