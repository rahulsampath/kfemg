
#include "gmg/include/lsFitPC.h"
#include "gmg/include/gmgUtils.h"

void setupLSfitPC1D(PC pc, Mat Kmat, Mat reducedMat, int K, int Nx,
    std::vector<long long int>& coeffsCK, std::vector<long long int>& coeffsC0) {
  LSfitData* data = new LSfitData;
  data->K = K;
  data->Nx = Nx;
  data->Kmat = Kmat;
  MatGetVecs(Kmat, &(data->err), &(data->res));
  VecDuplicate((data->res), &(data->g1Vec));
  VecDuplicate((data->res), &(data->g2Vec));
  VecDuplicate((data->res), &(data->tmp1));
  VecDuplicate((data->res), &(data->tmp2));
  MatGetVecs(reducedMat, &(data->reducedSol), &(data->reducedRhs));
  VecDuplicate((data->reducedRhs), &(data->reducedG1Vec));
  VecDuplicate((data->reducedRhs), &(data->reducedG2Vec));
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
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

  double* g1Arr;
  double* g2Arr;
  double* redG1Arr;
  double* redG2Arr;
  VecGetArray((data->g1Vec), &g1Arr);
  VecGetArray((data->reducedG1Vec), &redG1Arr);
  VecGetArray((data->g2Vec), &g2Arr);
  VecGetArray((data->reducedG2Vec), &redG2Arr);
  computeFxPhi1D(0, Nx, K, coeffsCK, g1Arr);
  computeFxPhi1D(0, Nx, 0, coeffsC0, redG1Arr);
  computeFxPhi1D(1, Nx, K, coeffsCK, g2Arr);
  computeFxPhi1D(1, Nx, 0, coeffsC0, redG2Arr);
  double Hmat[2][2];
  computeHmat(Hmat, (Nx*(K + 1)), g1Arr, g2Arr);
  VecRestoreArray((data->g1Vec), &g1Arr);
  VecRestoreArray((data->reducedG1Vec), &redG1Arr);
  VecRestoreArray((data->g2Vec), &g2Arr);
  VecRestoreArray((data->reducedG2Vec), &redG2Arr);
  matInvert2x2(Hmat, (data->HmatInv));

  PCShellSetContext(pc, data);
  PCShellSetName(pc, "MyLSPC");
  PCShellSetApply(pc, &applyLSfitPC1D);
  PCShellSetDestroy(pc, &destroyLSfitPC1D);
}

PetscErrorCode destroyLSfitPC1D(PC pc) {
  LSfitData* data;
  PCShellGetContext(pc, (void**)(&data));

  KSPDestroy(&(data->reducedSolver));
  VecDestroy(&(data->res));
  VecDestroy(&(data->err));
  VecDestroy(&(data->tmp1));
  VecDestroy(&(data->tmp2));
  VecDestroy(&(data->g1Vec));
  VecDestroy(&(data->g2Vec));
  VecDestroy(&(data->reducedG1Vec));
  VecDestroy(&(data->reducedG2Vec));
  VecDestroy(&(data->reducedRhs));
  VecDestroy(&(data->reducedSol));
  delete data;

  return 0;
}

PetscErrorCode applyLSfitPC1D(PC pc, Vec in, Vec out) {
  LSfitData* data;
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
    double* g1Arr;
    double* g2Arr;
    VecGetArray((data->res), &resArr);
    VecGetArray((data->g1Vec), &g1Arr);
    VecGetArray((data->g2Vec), &g2Arr);
    double aVec[2];
    computeLSfit(aVec, (data->HmatInv), ((data->Nx)*dofsPerNode), resArr, g1Arr, g2Arr);
    VecRestoreArray((data->res), &resArr);
    VecRestoreArray((data->g1Vec), &g1Arr);
    VecRestoreArray((data->g2Vec), &g2Arr);

    //4. Compute RHS for reduced problem rhs = a0*g1 + a1*g2
    double* rhsArr;
    VecGetArray((data->reducedG1Vec), &g1Arr);
    VecGetArray((data->reducedG2Vec), &g2Arr);
    VecGetArray((data->reducedRhs), &rhsArr);
    for(int i = 0; i < (data->Nx); ++i) {
      rhsArr[i] = (aVec[0] * g1Arr[i]) + (aVec[1] * g2Arr[i]); 
    }//end i
    VecRestoreArray((data->reducedG1Vec), &g1Arr);
    VecRestoreArray((data->reducedG2Vec), &g2Arr);
    VecRestoreArray((data->reducedRhs), &rhsArr);

    //5. Solve (approx.) the reduced problem using zero initial guess.
    KSPSolve((data->reducedSolver), (data->reducedRhs), (data->reducedSol));

    //6. Set reducedSol as the 0th dof of err
    double* errArr;
    double* solArr;
    VecGetArray((data->reducedSol), &solArr);
    VecGetArray((data->err), &errArr);
    for(int i = 0; i < (data->Nx); ++i) {
      errArr[i*dofsPerNode] = solArr[i];
    }//end i
    VecRestoreArray((data->reducedSol), &solArr);

    //7. Use Finite Differencing to estimate the other dofs of err.
    for(int d = 1; d <= (data->K); ++d) {
      errArr[(0*dofsPerNode) + d] = -((3.0 * errArr[(0*dofsPerNode) + d - 1]) - (4.0 * errArr[(1*dofsPerNode) + d - 1])
          + errArr[(2*dofsPerNode) + d - 1])/4.0;
      for(int i = 1; i < ((data->Nx) - 1); ++i) {
        errArr[(i*dofsPerNode) + d] = (errArr[((i + 1)*dofsPerNode) + d - 1] - inArr[((i - 1)*dofsPerNode) + d - 1])/4.0;
      }//end i
      errArr[(((data->Nx) - 1)*dofsPerNode) + d] = ((3.0 * errArr[(((data->Nx) - 1)*dofsPerNode) + d - 1]) -
          (4.0 * errArr[(((data->Nx) - 2)*dofsPerNode) + d - 1]) + errArr[(((data->Nx) - 3)*dofsPerNode) + d - 1])/4.0;
    }//end d
    VecRestoreArray((data->err), &errArr);

    PetscScalar errNormSqr;
    VecDot((data->err), (data->err), &errNormSqr);
    if(errNormSqr < 1.0e-24) {
      VecCopy(in, out);
    } else {
      //8. tmp1 = Kmat * err 
      MatMult((data->Kmat), (data->err), (data->tmp1));

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
      }

      //10. Accept preconditioner only if it is converging 
      if(finalNormSqr < initNormSqr) {
        VecAXPY(out, -alpha, (data->err));
      } else {
        VecCopy(in, out);
      }
    }
  }

  return 0;
}

void computeFxPhi1D(int mode, int Nx, int K, std::vector<long long int>& coeffs, double* res) {
  long double hx = 1.0L/(static_cast<long double>(Nx - 1));

  PetscInt extraNumGpts = 0;
  PetscOptionsGetInt(PETSC_NULL, "-extraGptsFxPhi", &extraNumGpts, PETSC_NULL);
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
          double fn;
          if(mode == 0) {
            fn = 1.0;
          } else {
            fn = sin((static_cast<double>(mode))*__PI__*xg);
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

void computeLSfit(double aVec[2], double HmatInv[2][2], int len, double* fVec, double* g1Vec, double* g2Vec) {
  aVec[0] = 0;
  aVec[1] = 0;
  double rVal = computeRval(aVec, len, fVec, g1Vec, g2Vec); 
  const int maxNewtonIters = 100;
  for(int iter = 0; iter < maxNewtonIters; ++iter) {
    if(rVal < 1.0e-12) {
      break;
    }
    double jVec[2];
    computeJvec(jVec, aVec, len, fVec, g1Vec, g2Vec);
    if((fabs(jVec[0]) < 1.0e-12) && (fabs(jVec[1]) < 1.0e-12)) {
      break;
    }
    double step[2];
    matMult2x2(HmatInv, jVec, step);
    if((fabs(step[0]) < 1.0e-12) && (fabs(step[1]) < 1.0e-12)) {
      break;
    }
    double alpha = 1.0;
    double tmpVec[2];
    tmpVec[0] = aVec[0] - (alpha*step[0]);
    tmpVec[1] = aVec[1] - (alpha*step[1]);
    double tmpVal = computeRval(tmpVec, len, fVec, g1Vec, g2Vec);
    while(alpha > 1.0e-12) {
      if(tmpVal < rVal) {
        break;
      }
      alpha *= 0.1;
      tmpVec[0] = aVec[0] - (alpha*step[0]);
      tmpVec[1] = aVec[1] - (alpha*step[1]);
      tmpVal = computeRval(tmpVec, len, fVec, g1Vec, g2Vec);
    }
    if(tmpVal < rVal) {
      aVec[0] = tmpVec[0];
      aVec[1] = tmpVec[1];
      rVal = tmpVal;
    } else {
      break;
    }
  }//end iter
}

double computeRval(double aVec[2], int len, double* fVec, double* g1Vec, double* g2Vec) {
  double res = 0;
  for(int i = 0; i < len; ++i) {
    double val = fVec[i] - (g1Vec[i]*aVec[0]) - (g2Vec[i]*aVec[1]);
    res += (val*val);
  }//end i
  return res;
}

void computeJvec(double jVec[2], double aVec[2], int len, double* fVec, double* g1Vec, double* g2Vec) {
  jVec[0] = 0;
  jVec[1] = 0;
  for(int i = 0; i < len; ++i) {
    double scaling = 2.0*((g1Vec[i]*aVec[0]) + (g2Vec[i]*aVec[1]) - fVec[i]);
    jVec[0] += (scaling*g1Vec[i]);
    jVec[1] += (scaling*g2Vec[i]);
  }//end i
}

void computeHmat(double mat[2][2], int len, double* g1Vec, double* g2Vec) {
  double a = 0;
  double b = 0;
  double c = 0;
  for(int i = 0; i < len; ++i) {
    a += (g1Vec[i] * g1Vec[i]);
    c += (g2Vec[i] * g1Vec[i]);
    b += (g2Vec[i] * g2Vec[i]);
  }//end i
  mat[0][0] = 2.0*a;
  mat[0][1] = 2.0*c;
  mat[1][0] = mat[0][1];
  mat[1][1] = 2.0*b;
}


