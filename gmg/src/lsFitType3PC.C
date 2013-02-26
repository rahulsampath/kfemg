
#include "gmg/include/lsFitType3PC.h"
#include "gmg/include/gmgUtils.h"
#include <iostream>

void setupLSfitType3PC(PC pc, Mat Kmat, Mat reducedMat, int K, int Nx,
    std::vector<long long int>& coeffsCK, std::vector<long long int>& coeffsC0) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
  LSfitType3Data* data = new LSfitType3Data;
  data->K = K;
  data->Nx = Nx;
  data->coeffsCK = &coeffsCK;
  data->coeffsC0 = &coeffsC0; 
  data->Kmat = Kmat;
  MatGetVecs(Kmat, &(data->err), &(data->res));
  VecDuplicate((data->res), &(data->tmp1));
  VecDuplicate((data->res), &(data->tmp2));
  VecDuplicate((data->res), &(data->fTilde));
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
  PCShellSetApply(pc, &applyLSfitType3PC);
  PCShellSetDestroy(pc, &destroyLSfitType3PC);
} 

PetscErrorCode destroyLSfitType3PC(PC pc) {
  LSfitType3Data* data;
  PCShellGetContext(pc, (void**)(&data));
  KSPDestroy(&(data->reducedSolver));
  VecDestroy(&(data->res));
  VecDestroy(&(data->err));
  VecDestroy(&(data->fTilde));
  VecDestroy(&(data->tmp1));
  VecDestroy(&(data->tmp2));
  VecDestroy(&(data->reducedRhs));
  VecDestroy(&(data->reducedSol));
  delete data;
  return 0;
}

void computeFtilde(double xStar, int Nx, int K, std::vector<long long int>& coeffs, double* res) {
  long double hx = 1.0L/(static_cast<long double>(Nx - 1));

  PetscInt extraNumGpts = 0;
  PetscOptionsGetInt(PETSC_NULL, "-extraGptsFhat", &extraNumGpts, PETSC_NULL);
  int numGaussPts = (2*K) + 2 + extraNumGpts;
  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  std::vector<std::vector<std::vector<long double> > > shFnVals(2);
  for(int nd = 0; nd < 2; ++nd) {
    shFnVals[nd].resize(K + 1);
    for(int dof = 0; dof <= K; ++dof) {
      (shFnVals[nd][dof]).resize(numGaussPts);
      for(int g = 0; g < numGaussPts; ++g) {
        shFnVals[nd][dof][g] = eval1DshFn(nd, dof, K, coeffs, gPt[g]);
      }//end g
    }//end dof
  }//end nd

  int dofsPerNode = K + 1;
  for(int i = 0; i < (dofsPerNode*Nx); ++i) {
    res[i] = 0.0;
  }//end i

  for(int xi = 0; xi < (Nx - 1); ++xi) {
    long double xa = (static_cast<long double>(xi))*hx;
    for(int nd = 0; nd < 2; ++nd) {
      for(int dof = 0; dof <= K; ++dof) {
        for(int g = 0; g < numGaussPts; ++g) {
          long double xg = coordLocalToGlobal(gPt[g], xa, hx);
          double fn = 0.0;
          if((fabs(xg - xStar)) < hx) {
            double denom = ((xg - xStar)*(xg - xStar)) - (hx*hx);
            fn = exp(-1.0/denom);
          }
          res[((xi + nd)*dofsPerNode) + dof] += ( gWt[g] * shFnVals[nd][dof][g] * fn );
        }//end g
      }//end dof
    }//end nd
  }//end xi

  long double jac = hx * 0.5L;
  for(int i = 0; i < (dofsPerNode*Nx); ++i) {
    res[i] *= jac;
  }//end i

  //Dirichlet Correction
  res[0] = 0;
  res[dofsPerNode*(Nx - 1)] = 0;
}

double computeRval(int len, double A, double* fVec, double* fTildeVec) {
  double res = 0.0;
  for(int i = 0; i < len; ++i) {
    double val = (A*fTildeVec[i]) - fVec[i];
    res += (val * val);
  }//end i
  return res;
}

double computeGradR(int len, double A, double* fVec, double* fTildeVec) {
  double res = 0.0;
  for(int i = 0; i < len; ++i) {
    res += (((A*fTildeVec[i]) - fVec[i])*fTildeVec[i]);
  }//end i
  res *= 2.0;
  return res;
}

double computeHessR(int len, double* fTildeVec) {
  double res = 0.0;
  for(int i = 0; i < len; ++i) {
    res += (fTildeVec[i]*fTildeVec[i]);
  }//end i
  res *= 2.0;
  return res;
}



