
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
  const VecType type;
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

  //This approximately solves: Kmat * out = in
  int dofsPerNode = (data->K) + 1;

  //1. Prepare initial guess that satisfies Dirichlet conditions.  
  VecZeroEntries(out);
  double* outArr;
  double* inArr;
  VecGetArray(out, &outArr);
  VecGetArray(in, &inArr);
  outArr[0] = inArr[0];
  outArr[((data->Nx) - 1)*dofsPerNode] = inArr[((data->Nx) - 1)*dofsPerNode];
  VecRestoreArray(in, &inArr);
  VecRestoreArray(out, &outArr);

  //2.a. Compute initial residual: res = in - (Kmat * out)
  computeResidual((data->Kmat), out, in, (data->res));
  //2.b. Initial residual norm.
  PetscScalar initNormSqr;
  VecDot((data->res), (data->res), &initNormSqr);

  if(initNormSqr >= 1.0e-24) {
    //3. Compute LS fit
    double* resArr;
    double* buf1;
    double* buf2;
    VecGetArray((data->res), &resArr);
    VecGetArray((data->fHat), &buf1);
    VecGetArray((data->gradFhat), &buf2);
    double yVec[3];
    double fit = computeLSfit(yVec, (data->Nx), (data->K), *(data->coeffsCK), resArr, buf1, buf2);
    VecRestoreArray((data->fHat), &buf1);
    VecRestoreArray((data->gradFhat), &buf2);
    VecRestoreArray((data->res), &resArr);
    std::cout<<"xStar = "<<(yVec[0])<<", sigma = "<<(yVec[1])<<", A = "<<(yVec[2])
      <<", fit = "<<fit<<", base = "<<initNormSqr<<std::endl;

    //4. Compute RHS for reduced problem 
    double* rhsArr;
    VecGetArray((data->reducedRhs), &rhsArr);
    computeFhat(yVec[0], yVec[1], yVec[2], (data->Nx), 0, *(data->coeffsC0), rhsArr);
    VecRestoreArray((data->reducedRhs), &rhsArr);

    //5. Solve (approx.) the reduced problem using zero initial guess.
    KSPSolve((data->reducedSolver), (data->reducedRhs), (data->reducedSol));

    //6. Set reducedSol as the 0th dof of err
    double* errArr;
    double* solArr;
    VecGetArray((data->err), &errArr);
    VecGetArray((data->reducedSol), &solArr);
    for(int i = 0; i < (data->Nx); ++i) {
      errArr[i*dofsPerNode] = solArr[i];
    }//end i
    VecRestoreArray((data->reducedSol), &solArr);

    //7. Use Finite Differencing to estimate the other dofs of err.
    PetscInt fdType = 2;
    PetscOptionsGetInt(PETSC_NULL, "-fdType", &fdType, PETSC_NULL);
    if(fdType == 1) {
      //Second Order
      for(int d = 1; d <= (data->K); ++d) {
        errArr[(0*dofsPerNode) + d] = -((3.0 * errArr[(0*dofsPerNode) + d - 1]) - (4.0 * errArr[(1*dofsPerNode) + d - 1])
            + errArr[(2*dofsPerNode) + d - 1])/4.0;
        for(int i = 1; i < ((data->Nx) - 1); ++i) {
          errArr[(i*dofsPerNode) + d] = (errArr[((i + 1)*dofsPerNode) + d - 1] - errArr[((i - 1)*dofsPerNode) + d - 1])/4.0;
        }//end i
        errArr[(((data->Nx) - 1)*dofsPerNode) + d] = ((3.0 * errArr[(((data->Nx) - 1)*dofsPerNode) + d - 1]) -
            (4.0 * errArr[(((data->Nx) - 2)*dofsPerNode) + d - 1]) + errArr[(((data->Nx) - 3)*dofsPerNode) + d - 1])/4.0;
      }//end d
    } else {
      //Fourth Order
      for(int d = 1; d <= (data->K); ++d) {
        errArr[(0*dofsPerNode) + d] = -((25.0 * errArr[(0*dofsPerNode) + d - 1]) -
            (48.0 * errArr[(1*dofsPerNode) + d - 1]) + (36.0 * errArr[(2*dofsPerNode) + d - 1])
            - (16.0 * errArr[(3*dofsPerNode) + d - 1]) +
            (3.0 * errArr[(4*dofsPerNode) + d - 1]))/24.0;
        errArr[(1*dofsPerNode) + d] = -((25.0 * errArr[(1*dofsPerNode) + d - 1]) -
            (48.0 * errArr[(2*dofsPerNode) + d - 1]) + (36.0 * errArr[(3*dofsPerNode) + d - 1])
            - (16.0 * errArr[(4*dofsPerNode) + d - 1]) +
            (3.0 * errArr[(5*dofsPerNode) + d - 1]))/24.0;
        for(int i = 2; i < ((data->Nx) - 2); ++i) {
          errArr[(i*dofsPerNode) + d] = (-errArr[((i + 2)*dofsPerNode) + d - 1] +
              (8.0 * errArr[((i + 1)*dofsPerNode) + d - 1]) - (8.0 * errArr[((i - 1)*dofsPerNode) + d - 1])
              + errArr[((i - 2)*dofsPerNode) + d - 1])/24.0;
        }//end i
        errArr[(((data->Nx) - 2)*dofsPerNode) + d] = ((25.0 * errArr[(((data->Nx) - 2)*dofsPerNode) + d - 1]) -
            (48.0 * errArr[(((data->Nx) - 3)*dofsPerNode) + d - 1]) +
            (36.0 * errArr[(((data->Nx) - 4)*dofsPerNode) + d - 1]) -
            (16.0 * errArr[(((data->Nx) - 5)*dofsPerNode) + d - 1]) +
            (3.0 * errArr[(((data->Nx) - 6)*dofsPerNode) + d - 1]))/24.0;
        errArr[(((data->Nx) - 1)*dofsPerNode) + d] = ((25.0 * errArr[(((data->Nx) - 1)*dofsPerNode) + d - 1]) -
            (48.0 * errArr[(((data->Nx) - 2)*dofsPerNode) + d - 1]) + 
            (36.0 * errArr[(((data->Nx) - 3)*dofsPerNode) + d - 1]) -
            (16.0 * errArr[(((data->Nx) - 4)*dofsPerNode) + d - 1]) +
            (3.0 * errArr[(((data->Nx) - 5)*dofsPerNode) + d - 1]))/24.0;
      }//end d
    }
    VecRestoreArray((data->err), &errArr);

    PetscScalar errNormSqr;
    VecDot((data->err), (data->err), &errNormSqr);
    if(errNormSqr < 1.0e-24) {
      std::cout<<"Rejected PC"<<std::endl;
      VecCopy(in, out);
    } else {
      //8. tmp1 = Kmat * err 
      MatMult((data->Kmat), (data->err), (data->tmp1));
      //    PetscReal acceptTol = 1.0;
      //   PetscOptionsGetReal(PETSC_NULL, "-acceptPCtol", &acceptTol, PETSC_NULL);
      //9. Simple line search
      double alpha = -1.0;
      VecWAXPY((data->tmp2), alpha, (data->tmp1), (data->res));
      PetscScalar finalNormSqr;
      VecDot((data->tmp2), (data->tmp2), &finalNormSqr);
      while(alpha < -1.0e-12) {
        if(finalNormSqr < initNormSqr) {
          break;
        }
        alpha *= 0.1;
        VecWAXPY((data->tmp2), alpha, (data->tmp1), (data->res));
        VecDot((data->tmp2), (data->tmp2), &finalNormSqr);
      }//end while
      std::cout<<"alpha = "<<alpha<<std::endl;
      //10. Accept preconditioner only if it is converging 
      if(finalNormSqr < initNormSqr) {
        std::cout<<"Accepted PC: init = "<<initNormSqr<<", final = "<<finalNormSqr<<std::endl;
        VecAXPY(out, -alpha, (data->err));
      } else {
        std::cout<<"Rejected PC"<<std::endl;
        VecCopy(in, out);
      }
    }
  }
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
  PetscInt maxIters = 100000;
  PetscOptionsGetInt(PETSC_NULL, "-maxOptIters", &maxIters, PETSC_NULL);
  int iter;
  for(iter = 0; iter < maxIters; ++iter) {
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
      std::cout<<"Line Search Failed!"<<std::endl;
      break;
    }
  }//end iter
  std::cout<<"Num Optimization Iters = "<<iter<<std::endl;
  return rVal;
}

void computeFhat(double xStar, double sigma, double A, int Nx, int K, 
    std::vector<long long int>& coeffs, double* res) {
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
          double val = (xg - xStar)/sigma;
          double fn = A * exp(-val * val);
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

void computeGradFhat(double xStar, double sigma, double A, int Nx, int K,
    std::vector<long long int>& coeffs, double* res) {
  long double hx = 1.0L/(static_cast<long double>(Nx - 1));

  PetscInt extraNumGpts = 0;
  PetscOptionsGetInt(PETSC_NULL, "-extraGptsGradFhat", &extraNumGpts, PETSC_NULL);
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
  for(int i = 0; i < (3*dofsPerNode*Nx); ++i) {
    res[i] = 0.0;
  }//end i

  for(int xi = 0; xi < (Nx - 1); ++xi) {
    long double xa = (static_cast<long double>(xi))*hx;
    for(int nd = 0; nd < 2; ++nd) {
      for(int d = 0; d <= K; ++d) {
        for(int g = 0; g < numGaussPts; ++g) {
          long double xg = coordLocalToGlobal(gPt[g], xa, hx);
          double lVal = (xg - xStar)/sigma;
          double hVal = -lVal*lVal;
          double gradL[2];
          gradL[0] = -1.0/sigma;
          gradL[1] = -lVal/sigma;
          double gradH[2];
          for(int k = 0; k < 2; ++k) {
            gradH[k] = -2.0 * lVal * gradL[k];
          }//end k
          double scale = exp(hVal);
          double gradG[3];
          for(int k = 0; k < 2; ++k) {
            gradG[k] = scale * A * gradH[k];
          }//end k
          gradG[2] = scale;
          for(int j = 0; j < 3; ++j) {
            int id = ((xi + nd)*dofsPerNode) + d;
            res[(3*id) + j] += ( gWt[g] * shFnVals[nd][d][g] * gradG[j] );
          }//end j
        }//end g
      }//end d
    }//end nd
  }//end xi

  long double jac = hx * 0.5L;
  for(int i = 0; i < (3*dofsPerNode*Nx); ++i) {
    res[i] *= jac;
  }//end i

  //Dirichlet Correction
  for(int j = 0; j < 3; ++j) {
    res[j] = 0;
    res[(3*dofsPerNode*(Nx - 1)) + j] = 0;
  }//end j
}

void computeGradRvec(double gradRvec[3], int len, double* fVec,
    double* fHatVec, double* gradFhatVec) {
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


