
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
  Vec res;
  Vec err;
  Vec tmp1;
  Vec tmp2;
  Vec g1Vec;
  Vec g2Vec;
  Vec reducedG1Vec;
  Vec reducedG2Vec;
  KSP reducedSolver;
  Vec reducedRhs;
  Vec reducedSol;
};

void computeFxPhi1D(int mode, int Nx, int K, std::vector<long long int>& coeffs, double* res);

void computeLSfit(double aVec[2], double HmatInv[2][2], int len, double* fVec, double* g1Vec, double* g2Vec);

double computeRval(double aVec[2], int len, double* fVec, double* g1Vec, double* g2Vec);

void computeJvec(double jVec[2], double aVec[2], int len, double* fVec, double* g1Vec, double* g2Vec);

void computeHmat(double mat[2][2], int len, double* g1Vec, double* g2Vec);

void applyLSfitPC1D(LSfitData* data, Vec in, Vec out);

#endif

