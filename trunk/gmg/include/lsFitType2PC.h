
#ifndef __LS_FIT_TYPE2_PC__
#define __LS_FIT_TYPE2_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include <vector>

struct LSfitType2Data {
  int K;
  int Nx;
  std::vector<long long int>* coeffsCK;
  std::vector<long long int>* coeffsC0;
  Mat Kmat;
  KSP reducedSolver;
  Vec fHat;
  Vec gradFhat;
  Vec res;
  Vec err;
  Vec tmp1;
  Vec tmp2;
  Vec reducedRhs;
  Vec reducedSol;
};

void computeFhat(double xStar, double sigma, double A, int Nx, int K, 
    std::vector<long long int>& coeffs, double* res);

void computeGradFhat(double xStar, double sigma, double A, int Nx, int K,
    std::vector<long long int>& coeffs, double* res);

double computeLSfit(double yVec[3], int len, double* fVec);

double computeRval(int len, double* fVec, double* fHatVec);

void computeGradRvec(double gradRvec[3], int len, double* fVec, double* fHatVec, double* gradFhatVec);

PetscErrorCode applyLSfitType2PC(PC pc, Vec in, Vec out);

PetscErrorCode destroyLSfitType2PC(PC pc);

void setupLSfitType2PC(PC pc, Mat Kmat, Mat reducedMat, int K, int Nx,
    std::vector<long long int>& coeffsCK, std::vector<long long int>& coeffsC0);

#endif


