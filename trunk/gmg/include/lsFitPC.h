
#ifndef __LS_FIT_PC__
#define __LS_FIT_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include <vector>

struct LSfitData {
  int K;
  int Nx;
  double HmatInv[2][2];
  Mat Kmat;
  KSP reducedSolver;
  Vec res;
  Vec err;
  Vec tmp1;
  Vec tmp2;
  Vec g1Vec;
  Vec g2Vec;
  Vec reducedG1Vec;
  Vec reducedG2Vec;
  Vec reducedRhs;
  Vec reducedSol;
};

void computeFxPhi1D(int mode, int Nx, int K, std::vector<long long int>& coeffs, double* res);

void computeLSfit(double aVec[2], double HmatInv[2][2], int len, double* fVec, double* g1Vec, double* g2Vec);

double computeRval(double aVec[2], int len, double* fVec, double* g1Vec, double* g2Vec);

void computeJvec(double jVec[2], double aVec[2], int len, double* fVec, double* g1Vec, double* g2Vec);

void computeHmat(double mat[2][2], int len, double* g1Vec, double* g2Vec);

PetscErrorCode applyLSfitPC1D(PC pc, Vec in, Vec out);

PetscErrorCode destroyLSfitPC1D(PC pc);

void createLSfitPC1D(PC pc, Mat Kmat, Mat reducedMat, int K, int Nx,
    std::vector<long long int>& coeffsK, std::vector<long long int>& coeffs0);

#endif


