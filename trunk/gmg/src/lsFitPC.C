

void computeFxPhi1D(int mode, int Nx, int K, std::vector<long long int>& coeffs,
    std::vector<double>& res) {
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

  res.clear();
  res.resize((dofsPerNode*Nx), 0.0);

  for(PetscInt xi = 0; xi < (Nx - 1); ++xi) {
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
  for(size_t i = 0; i < res.size(); ++i) {
    res[i] *= jac;
  }//end i

  res[0] = 0;
  res[dofsPerNode*(Nx - 1)] = 0;
}

void computeLSfit(double aVec[2], double HmatInv[2][2], std::vector<double>& fVec,
    std::vector<double>& gVec, std::vector<double>& cVec) {
  aVec[0] = 0;
  aVec[1] = 0;
  double rVal = computeRval(aVec, fVec, gVec, cVec); 
  const int maxNewtonIters = 100;
  for(int iter = 0; iter < maxNewtonIters; ++iter) {
    if(rVal < 1.0e-12) {
      break;
    }
    double jVec[2];
    computeJvec(jVec, aVec, fVec, gVec, cVec);
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
    double tmpVal = computeRval(tmpVec, fVec, gVec, cVec);
    while(alpha > 1.0e-12) {
      if(tmpVal < rVal) {
        break;
      }
      alpha *= 0.1;
      tmpVec[0] = aVec[0] - (alpha*step[0]);
      tmpVec[1] = aVec[1] - (alpha*step[1]);
      tmpVal = computeRval(tmpVec, fVec, gVec, cVec);
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

double computeRval(double aVec[2], std::vector<double>& fVec, std::vector<double>& gVec, 
    std::vector<double>& cVec) {
  double res = 0;
  for(size_t i = 0; i < fVec.size(); ++i) {
    double val = fVec[i] - (gVec[i]*aVec[0]) - (cVec[i]*aVec[1]);
    res += (val*val);
  }//end i
  return res;
}

void computeJvec(double jVec[2], double aVec[2], std::vector<double>& fVec,
    std::vector<double>& gVec, std::vector<double>& cVec) {
  jVec[0] = 0;
  jVec[1] = 0;
  for(size_t i = 0; i < fVec.size(); ++i) {
    double scaling = 2.0*((gVec[i]*aVec[0]) + (cVec[i]*aVec[1]) - fVec[i]);
    jVec[0] += (scaling*gVec[i]);
    jVec[1] += (scaling*cVec[i]);
  }//end i
}

void computeHmat(double mat[2][2], std::vector<double>& gVec, std::vector<double>& cVec) {
  double a = 0;
  double b = 0;
  double c = 0;
  for(size_t i = 0; i < gVec.size(); ++i) {
    a += (gVec[i] * gVec[i]);
    c += (cVec[i] * gVec[i]);
    b += (cVec[i] * cVec[i]);
  }//end i
  mat[0][0] = 2.0*a;
  mat[0][1] = 2.0*c;
  mat[1][0] = mat[0][1];
  mat[1][1] = 2.0*b;
}


