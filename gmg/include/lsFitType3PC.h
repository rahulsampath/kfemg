
#ifndef __LS_FIT_TYPE3_PC__
#define __LS_FIT_TYPE3_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include <vector>

struct LSfitType3Data {
  int K;
  int Nx;
  std::vector<long long int>* coeffsCK;
  std::vector<long long int>* coeffsC0;
  Mat Kmat;
  KSP reducedSolver;
  Vec fTilde;
  Vec res;
  Vec err;
  Vec tmp1;
  Vec tmp2;
  Vec reducedRhs;
  Vec reducedSol;
};

void computeFtilde(double xStar, int Nx, int K, std::vector<long long int>& coeffs, double* res);

double computeRval(int len, double A, double* fVec, double* fTildeVec);

double computeLSfit(double& A, double xStar, int Nx, int K, std::vector<long long int>& coeffs,
    double* fVec, double* fTildeVec);

PetscErrorCode applyLSfitType3PC(PC pc, Vec in, Vec out);

PetscErrorCode destroyLSfitType3PC(PC pc);

void setupLSfitType3PC(PC pc, Mat Kmat, Mat reducedMat, int K, int Nx,
    std::vector<long long int>& coeffsCK, std::vector<long long int>& coeffsC0);

#endif



