
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

PetscErrorCode applyLSfitType3PC(PC pc, Vec in, Vec out) {
  LSfitType3Data* data;
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
    double* buf;
    VecGetArray((data->res), &resArr);
    VecGetArray((data->fTilde), &buf);
    long double hx = 1.0L/(static_cast<long double>((data->Nx) - 1));
    int iStar = 0;
    double maxVal = fabs(resArr[0]);
    for(int i = 0; i < (data->Nx); ++i) {
      for(int d = 0; d < dofsPerNode; ++d) {
        double val = fabs(resArr[(i*dofsPerNode) + d]);
        if(val > maxVal) {
          maxVal = val;
          iStar = i;
        }
      }//end d
    }//end i
    double xStar = (static_cast<double>(iStar))*hx;
    double A;
    double fit = computeLSfit(A, iStar, (data->Nx), (data->K), *(data->coeffsCK), resArr, buf);
    //std::cout<<"iStar = "<<iStar<<" xStar = "<<xStar<<" A = "<<A<<std::endl;
    /*
       for(int i = iStar - 1; i <= iStar + 1; ++i) {
       if(i < 0) {
       continue;
       }
       if(i >= (data->Nx)) {
       continue;
       }
       for(int d = 0; d < dofsPerNode; ++d) {
       int j = (i*dofsPerNode) + d;
       double err = fabs(resArr[j] - (A * buf[j]));
       std::cout<<"@ i = "<<i<<" d = "<<d<<" err = "<<err
       <<" f = "<<(resArr[j])<<" fTilde = "<<(buf[j])<<std::endl; 
       }//end d
       }//end i
       */
    VecRestoreArray((data->fTilde), &buf);
    VecRestoreArray((data->res), &resArr);

    //4. Compute RHS for reduced problem 
    double* rhsArr;
    VecGetArray((data->reducedRhs), &rhsArr);
    computeFtilde(iStar, (data->Nx), 0, *(data->coeffsC0), rhsArr);
    VecRestoreArray((data->reducedRhs), &rhsArr);
    VecScale((data->reducedRhs), A);

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
      /*
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
      */
      // std::cout<<"alpha = "<<alpha<<std::endl;
      //10. Accept preconditioner only if it is converging a
      /*
      if(finalNormSqr < initNormSqr) {
        //   std::cout<<"Accepted PC: init = "<<initNormSqr<<", final = "<<finalNormSqr<<std::endl;
        VecAXPY(out, -alpha, (data->err));
      } else {
        std::cout<<"Rejected PC"<<std::endl;
        VecCopy(in, out);
      }
      */
        VecAXPY(out, -alpha, (data->err));
    }
  }
  return 0;
}

double computeLSfit(double& A, int iStar, int Nx, int K, std::vector<long long int>& coeffs,
    double* fVec, double* fTildeVec) {
  int dofsPerNode = (K + 1);
  computeFtilde(iStar, Nx, K, coeffs, fTildeVec);
  double hessR = computeHessR(iStar, Nx, dofsPerNode, fTildeVec);
  A = 0.0;
  double rVal = computeRval(iStar, Nx, dofsPerNode, A, fVec, fTildeVec);
  PetscInt maxIters = 100;
  PetscOptionsGetInt(PETSC_NULL, "-maxOptIters", &maxIters, PETSC_NULL);
  int iter;
  for(iter = 0; iter < maxIters; ++iter) {
    //std::cout<<"R = "<<rVal<<std::endl;
    if(rVal < 1.0e-12) {
      //std::cout<<"R is zero!"<<std::endl;
      break;
    }
    double gradR = computeGradR(iStar, Nx, dofsPerNode, A, fVec, fTildeVec);
    //std::cout<<"GradR = "<<gradR<<std::endl;
    if((fabs(gradR)) < 1.0e-12) {
      //std::cout<<"gradR is zero!"<<std::endl;
      break;
    }
    double alpha = 1.0;
    double tmpA = A - (alpha * (gradR/hessR));
    double tmpVal = computeRval(iStar, Nx, dofsPerNode, tmpA, fVec, fTildeVec);
    while(alpha > 1.0e-12) {
      if(tmpVal < rVal) {
        break;
      }
      alpha *= 0.1;
      tmpA = A - (alpha * (gradR/hessR));
      tmpVal = computeRval(iStar, Nx, dofsPerNode, tmpA, fVec, fTildeVec);
    }//end while
    if(tmpVal < rVal) {
      A = tmpA;
      rVal = tmpVal;
    } else {
      //std::cout<<"Line Search Failed!"<<std::endl;
      break;
    }
  }//end iter
  //std::cout<<"Num Optimization Iters = "<<iter<<std::endl;
  return rVal;
}

void computeFtilde(int iStar, int Nx, int K, std::vector<long long int>& coeffs, double* res) {
  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  double xStar = (static_cast<double>(iStar)) * hx;

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
            double numer = -(hx*hx)*log(1.0);
            double denom = ((xg - xStar)*(xg - xStar)) - (hx*hx);
            fn = exp(numer/denom);
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

double computeRval(int iStar, int Nx, int dofsPerNode,
    double A, double* fVec, double* fTildeVec) {
  double res = 0.0;
  for(int i = (iStar - 1); i <= (iStar + 1); ++i) {
    if(i < 0) {
      continue;
    }
    if(i >= Nx) {
      continue;
    }
    for(int d = 0; d < dofsPerNode; ++d) {
      int j = (i*dofsPerNode) + d;
      double val = (A*fTildeVec[j]) - fVec[j];
      res += (val * val);
    }//end d
  }//end i
  return res;
}

double computeGradR(int iStar, int Nx, int dofsPerNode,
    double A, double* fVec, double* fTildeVec) {
  double res = 0.0;
  for(int i = (iStar - 1); i <= (iStar + 1); ++i) {
    if(i < 0) {
      continue;
    }
    if(i >= Nx) {
      continue;
    }
    for(int d = 0; d < dofsPerNode; ++d) {
      int j = (i*dofsPerNode) + d;
      res += (((A*fTildeVec[j]) - fVec[j])*fTildeVec[j]);
    }//end d
  }//end i
  res *= 2.0;
  return res;
}

double computeHessR(int iStar, int Nx, int dofsPerNode, double* fTildeVec) {
  double res = 0.0;
  for(int i = (iStar - 1); i <= (iStar + 1); ++i) {
    if(i < 0) {
      continue;
    }
    if(i >= Nx) {
      continue;
    }
    for(int d = 0; d < dofsPerNode; ++d) {
      int j = (i*dofsPerNode) + d;
      res += (fTildeVec[j]*fTildeVec[j]);
    }//end d
  }//end i
  res *= 2.0;
  return res;
}



