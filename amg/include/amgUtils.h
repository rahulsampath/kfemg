
#ifndef __AMG_UTILS__
#define __AMG_UTILS__

#include <vector>
#include "ml_include.h"

struct MyMatrix {
  std::vector<std::vector<unsigned int> > nzCols;
  std::vector<std::vector<long double> > vals;
};

void zeroBoundaries(double* arr, const unsigned int K, const unsigned int dim,
    const int Nz, const int Ny, const int Nx);

void printMatrix(MyMatrix & myMat);

void getNeighbors(std::vector<int> & nh, int zi, int yi, int xi, int Nz, int Ny, int Nx);

void assembleMatrix(MyMatrix & myMat, std::vector<std::vector<long double> > const & elemMat, const unsigned int K, 
    const unsigned int dim, const unsigned int Nz, const unsigned int Ny, const unsigned int Nx);

void dirichletMatrixCorrection(MyMatrix & myMat, const unsigned int K, const unsigned int dim,
    const int Nz, const int Ny, const int Nx);

void setValue(MyMatrix & myMat, unsigned int row, unsigned int col, long double val);

int myMatVec(ML_Operator* data, int in_length, double in[], int out_length, double out[]);

void myMatVecPrivate(MyMatrix* myMat, const unsigned int len, double* in, double* out);

int myGetRow(ML_Operator* data, int N_requested_rows, int requested_rows[],
    int allocated_space, int columns[], double values[], int row_lengths[]);

void createMLobjects(ML*& ml_obj, ML_Aggregate*& agg_obj, const unsigned int numGrids, const unsigned int maxIters,
    const double rTol, const unsigned int dim, const unsigned int K, MyMatrix& myMat);

void destroyMLobjects(ML*& ml_obj, ML_Aggregate*& agg_obj);

void computeRandomRHS(double* rhsArr, MyMatrix & myMat, const unsigned int K, const unsigned int dim,
    const int Nz, const int Ny, const int Nx);

void createKrylovObject(ML_Krylov*& krylov_obj, ML* ml_obj, const unsigned int maxIters, const double rTol);

#endif


