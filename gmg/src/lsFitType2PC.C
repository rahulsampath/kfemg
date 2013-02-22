
#include "gmg/include/lsFitType2PC.h"
#include "gmg/include/gmgUtils.h"
#include <iostream>

void setupLSfitType2PC(PC pc, Mat Kmat, Mat reducedMat, int K, int Nx,
    std::vector<long long int>& coeffsCK, std::vector<long long int>& coeffsC0) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
  LSfitType2Data* data = new LSfitType2Data;
  data->K = K;
  data->Nx = Nx;
  data->coeffsCK = &coeffsCK;
  data->coeffsC0 = &coeffsC0; 
  data->Kmat = Kmat;
  MatGetVecs(Kmat, &(data->err), &(data->res));
  VecDuplicate((data->res), &(data->tmp1));
  VecDuplicate((data->res), &(data->tmp2));
  VecDuplicate((data->res), &(data->fHat));
  VecCreate(comm, &(data->gradFhat));
  PetscInt gSz;
  PetscInt lSz;
  VecType type;
  VecGetType((data->res), &type);
  VecSetType((data->gradFhat), type);
  VecGetSize((data->res), &gSz);
  VecGetLocalSize((data->res), &lSz);
  VecSetSizes((data->gradFhat), (3*lSz), (3*gSz));
  MatGetVecs(reducedMat, &(data->reducedSol), &(data->reducedRhs));
  KSPCreate(comm, &(data->reducedSolver));
  KSPSetType((data->reducedSolver), KSPCG);
  KSPSetPCSide((data->reducedSolver), PC_LEFT);
  PC tmpPC;
  KSPGetPC((data->reducedSolver), &tmpPC);
  PCSetType(tmpPC, PCNONE);
  KSPSetOperators((data->reducedSolver), reducedMat, reducedMat, SAME_PRECONDITIONER);
  KSPSetInitialGuessNonzero((data->reducedSolver), PETSC_FALSE);
  KSPSetTolerances((data->reducedSolver), 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
  KSPSetOptionsPrefix((data->reducedSolver), "C0_");
  KSPSetFromOptions(data->reducedSolver);
  PCShellSetContext(pc, data);
  PCShellSetName(pc, "MyLSPC");
  PCShellSetApply(pc, &applyLSfitType2PC);
  PCShellSetDestroy(pc, &destroyLSfitType2PC);
} 

PetscErrorCode destroyLSfitType2PC(PC pc) {
  LSfitType2Data* data;
  PCShellGetContext(pc, (void**)(&data));
  KSPDestroy(&(data->reducedSolver));
  VecDestroy(&(data->res));
  VecDestroy(&(data->err));
  VecDestroy(&(data->fHat));
  VecDestroy(&(data->gradFhat));
  VecDestroy(&(data->tmp1));
  VecDestroy(&(data->tmp2));
  VecDestroy(&(data->reducedRhs));
  VecDestroy(&(data->reducedSol));
  delete data;
  return 0;
}

PetscErrorCode applyLSfitType2PC(PC pc, Vec in, Vec out) {
  LSfitType2Data* data;
  PCShellGetContext(pc, (void**)(&data));
  return 0;
}

double computeLSfit(double yVec[3], int Nx, int K, std::vector<long long int>& coeffs,
    double* fVec, double* fHatVec, double* gradFhatVec) {
  const int len = Nx*(K + 1);
  yVec[0] = 0.5;
  yVec[1] = 1.0;
  yVec[2] = 0.0;
  computeFhat(yVec[0], yVec[1], yVec[2], Nx, K, coeffs, fHatVec);
  double rVal = computeRval(len, fVec, fHatVec);
  const int maxIters = 100;
  for(int iter = 0; iter < maxIters; ++iter) {
    if(rVal < 1.0e-12) {
      break;
    }
    computeGradFhat(yVec[0], yVec[1], yVec[2], Nx, K, coeffs, gradFhatVec);
    double gradRvec[3];
    computeGradRvec(gradRvec, len, fVec, fHatVec, gradFhatVec);
    if( (fabs(gradRvec[0]) < 1.0e-12) &&
        (fabs(gradRvec[1]) < 1.0e-12) &&
        (fabs(gradRvec[2]) < 1.0e-12) ) {
      break;
    }
    double alpha = 1.0;
    double tmpVec[3];
    for(int j = 0; j < 3; ++j) {
      tmpVec[j] = yVec[j] - (alpha * gradRvec[j]);
    }//end j
    computeFhat(tmpVec[0], tmpVec[1], tmpVec[2], Nx, K, coeffs, fHatVec);
    double tmpVal = computeRval(len, fVec, fHatVec);
    while(alpha > 1.0e-12) {
      if(tmpVal < rVal) {
        break;
      }
      alpha *= 0.1;
      for(int j = 0; j < 3; ++j) {
        tmpVec[j] = yVec[j] - (alpha * gradRvec[j]);
      }//end j
      computeFhat(tmpVec[0], tmpVec[1], tmpVec[2], Nx, K, coeffs, fHatVec);
      tmpVal = computeRval(len, fVec, fHatVec);
    }//end while
    if(tmpVal < rVal) {
      for(int j = 0; j < 3; ++j) {
        yVec[j] = tmpVec[j];
      }//end j
      rVal = tmpVal;
    } else {
      break;
    }
  }//end iter
  return rVal;
}

void computeFhat(double xStar, double sigma, double A, int Nx, int K, 
    std::vector<long long int>& coeffs, double* res) {
}

void computeGradFhat(double xStar, double sigma, double A, int Nx, int K,
    std::vector<long long int>& coeffs, double* res) {
}

void computeGradRvec(double gradRvec[3], int len, double* fVec, double* fHatVec, double* gradFhatVec) {
  for(int j = 0; j < 3; ++j) {
    gradRvec[j] = 0.0;
  }//end j
  for(int i = 0; i < len; ++i) {
    double scale = 2.0*(fHatVec[i] - fVec[i]);
    for(int j = 0; j < 3; ++j) {
      gradRvec[j] += (scale * gradFhatVec[(3*i) + j]);
    }//end j
  }//end i
}

double computeRval(int len, double* fVec, double* fHatVec) {
  double res = 0.0;
  for(int i = 0; i < len; ++i) {
    double val = (fHatVec[i] - fVec[i]);
    res += (val * val);
  }//end i
  return res;
}


