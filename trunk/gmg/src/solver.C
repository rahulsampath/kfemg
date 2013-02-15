
#include <vector>
#include <iostream>
#include <cmath>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

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

void createAll1DhatPc(std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<std::vector<Mat> > >& blkKmats,
    std::vector<std::vector<Mat> >& Khat1Dmats, std::vector<std::vector<PC> >& hatPc) {
  hatPc.resize(Khat1Dmats.size());
  for(int i = 0; i < (Khat1Dmats.size()); ++i) {
    hatPc[i].resize(Khat1Dmats[i].size());
    for(int j = 0; j < (Khat1Dmats[i].size()); ++j) {
      MPI_Comm comm;
      PetscObjectGetComm((PetscObject)(Khat1Dmats[i][j]), &comm);
      PCCreate(comm, &(hatPc[i][j]));
      PCSetType(hatPc[i][j], PCSHELL);
      PCFD1Ddata* data = new PCFD1Ddata; 
      PCShellSetContext(hatPc[i][j], data);
      PCShellSetName(hatPc[i][j], "MyPCFD");
      PCShellSetApply(hatPc[i][j], &applyPCFD1D);
      KSP ksp;
      KSPCreate(comm, &ksp);
      KSPSetType(ksp, KSPFGMRES);
      KSPSetPCSide(ksp, PC_RIGHT);
      if(j == 0) {
        PC pc;
        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCNONE);
        KSPSetOptionsPrefix(ksp, "hat0_");
      } else {
        KSPSetPC(ksp, hatPc[i][j - 1]);
        KSPSetOptionsPrefix(ksp, "hat1_");
      }
      KSPSetOperators(ksp, Khat1Dmats[i][j], Khat1Dmats[i][j], SAME_PRECONDITIONER);
      KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
      KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetFromOptions(ksp);
      data->ksp = ksp;
      data->partX = &(partX[i + 1]);
      data->numDofs = j + 2;
      MatGetVecs(Khat1Dmats[i][j], &(data->sol), &(data->rhs));
      MatGetVecs(blkKmats[i][0][0], &(data->u), &(data->uPrime));
    }//end j
  }//end i
}

void createAll1DmatShells(int K, std::vector<MPI_Comm>& activeComms, 
    std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<Mat> >& Khat1Dmats) {
  int nlevels = activeComms.size();
  Khat1Dmats.resize(nlevels - 1);
  for(int lev = 1; lev < nlevels; ++lev) {
    if(activeComms[lev] != MPI_COMM_NULL) {
      create1DmatShells(activeComms[lev], K, blkKmats[lev - 1], partX[lev], Khat1Dmats[lev - 1]); 
    }
  }//end lev
}

void create1DmatShells(MPI_Comm comm, int K, std::vector<std::vector<Mat> >& blkKmats,
    std::vector<PetscInt>& partX, std::vector<Mat>& Khat1Dmats) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  int nx = partX[rank];
  Khat1Dmats.resize(K, NULL);
  for(int i = 0; i < K; ++i) {
    Khat1Ddata* hatData = new Khat1Ddata; 
    MatGetVecs(blkKmats[0][0], &(hatData->u), &(hatData->uPrime));
    MatGetVecs(blkKmats[0][0], PETSC_NULL, &(hatData->tmpOut));
    hatData->blkKmats = &blkKmats;
    hatData->partX = &partX;
    hatData->numDofs = i + 1;
    hatData->K = K;
    MatCreateShell(comm, (nx*(i + 1)), (nx*(i + 1)), PETSC_DETERMINE, PETSC_DETERMINE, hatData, &(Khat1Dmats[i]));
    MatShellSetOperation(Khat1Dmats[i], MATOP_MULT, (void(*)(void))(&Khat1Dmult));
  }//end i
}

PetscErrorCode applyMG(PC pc, Vec in, Vec out) {
  MGdata* data;
  PCShellGetContext(pc, (void**)(&data));

  int nlevels = (data->Kmat).size();
  VecZeroEntries(out);
  data->mgSol[data->Kmat.size() - 1] = out;
  data->mgRhs[data->Kmat.size() - 1] = in;
  for(int iter = 0; iter < data->numVcycles; ++iter) {
    applyVcycle((nlevels - 1), data->Kmat, data->Pmat, data->tmpCvec, data->smoother,
        data->coarseSolver, data->mgSol, data->mgRhs, data->mgRes);
  }//end iter
  data->mgSol[data->Kmat.size() - 1] = NULL;
  data->mgRhs[data->Kmat.size() - 1] = NULL;

  return 0;
}

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, 
    std::vector<Vec>& tmpCvec, std::vector<KSP>& smoother, KSP coarseSolver,
    std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes) {
  if(currLev == 0) {
    KSPSolve(coarseSolver, mgRhs[currLev], mgSol[currLev]);
  } else {
    KSPSolve(smoother[currLev - 1], mgRhs[currLev], mgSol[currLev]);
    computeResidual(Kmat[currLev], mgSol[currLev], mgRhs[currLev], mgRes[currLev]);
    applyRestriction(Pmat[currLev - 1], tmpCvec[currLev - 1], mgRes[currLev], mgRhs[currLev - 1]);
    if(mgSol[currLev - 1] != NULL) {
      VecZeroEntries(mgSol[currLev - 1]);
      applyVcycle((currLev - 1), Kmat, Pmat, tmpCvec, smoother,
          coarseSolver, mgSol, mgRhs, mgRes);
    }
    applyProlongation(Pmat[currLev - 1], tmpCvec[currLev - 1], mgSol[currLev - 1], mgRes[currLev]);
    VecAXPY(mgSol[currLev], 1.0, mgRes[currLev]);
    KSPSolve(smoother[currLev - 1], mgRhs[currLev], mgSol[currLev]);
  }
}

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes) {
  mgSol.resize(Kmat.size(), NULL);
  mgRhs.resize(Kmat.size(), NULL);
  mgRes.resize(Kmat.size(), NULL);
  for(size_t i = 0; i < (Kmat.size() - 1); ++i) {
    if(Kmat[i] != NULL) {
      MatGetVecs(Kmat[i], &(mgSol[i]), &(mgRhs[i]));
      VecDuplicate(mgRhs[i], &(mgRes[i]));
    }
  }//end i
  MatGetVecs(Kmat[Kmat.size() - 1], NULL, &(mgRes[Kmat.size() - 1]));
}

PetscErrorCode Khat1Dmult(Mat mat, Vec in, Vec out) {
  Khat1Ddata* data;
  MatShellGetContext(mat, &data);

  MPI_Comm comm;
  PetscObjectGetComm((PetscObject)mat, &comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  int nx = (*(data->partX))[rank];
  int numDofs = data->numDofs;
  int K = data->K;

  VecZeroEntries(out);

  double* outArr;
  VecGetArray(out, &outArr);

  double* inArr;
  VecGetArray(in, &inArr);
  for(int c = 0; c < numDofs; ++c) {
    double* uArr;
    VecGetArray((data->u), &uArr);
    for(int i = 0; i < nx; ++i) {
      uArr[i] = inArr[(numDofs * i) + c];
    }//end i
    VecRestoreArray((data->u), &uArr);

    for(int r = 0; r < numDofs; ++r) {
      if(r <= c) {
        MatMult(((*(data->blkKmats))[r][c - r]), (data->u), (data->tmpOut));
      } else {
        MatMultTranspose(((*(data->blkKmats))[c][r - c]), (data->u), (data->tmpOut));
      }
      double* tmpArr;
      VecGetArray((data->tmpOut), &tmpArr);
      for(int i = 0; i < nx; ++i) {
        outArr[(numDofs * i) + r] += tmpArr[i];
      }//end i
      VecRestoreArray((data->tmpOut), &tmpArr);
    }//end r
  }//end c
  VecRestoreArray(in, &inArr);

  for(int c = numDofs; c <= K; ++c) {
    applyFD1D(comm, *(data->partX), (data->u), (data->uPrime));
    for(int r = 0; r < numDofs; ++r) {
      MatMult(((*(data->blkKmats))[r][c - r]), (data->uPrime), (data->tmpOut));
      double* tmpArr;
      VecGetArray((data->tmpOut), &tmpArr);
      for(int i = 0; i < nx; ++i) {
        outArr[(numDofs * i) + r] += tmpArr[i];
      }//end i
      VecRestoreArray((data->tmpOut), &tmpArr);
    }//end r
    VecSwap((data->u), (data->uPrime));
  }//end c

  VecRestoreArray(out, &outArr);

  return 0;
}

PetscErrorCode applyPCFD1D(PC pc, Vec in, Vec out) {
  PCFD1Ddata* data;
  PCShellGetContext(pc, (void**)(&data));

  MPI_Comm comm;
  PetscObjectGetComm((PetscObject)pc, &comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  int nx = (*(data->partX))[rank];
  int numDofs = data->numDofs;

  double* inArr;
  double* rhsArr;
  VecGetArray((data->rhs), &rhsArr);
  VecGetArray(in, &inArr);
  for(int i = 0; i < nx; ++i) {
    for(int d = 0; d < (numDofs - 1); ++d) {
      rhsArr[((numDofs - 1) * i) + d] = inArr[(numDofs * i) + d];
    }//end d
  }//end i
  VecRestoreArray((data->rhs), &rhsArr);
  VecRestoreArray(in, &inArr);

  KSPSolve((data->ksp), (data->rhs), (data->sol));

  double* outArr;
  VecGetArray(out, &outArr);

  double* solArr;
  double* uArr;
  VecGetArray((data->sol), &solArr);
  VecGetArray((data->u), &uArr);
  for(int i = 0; i < nx; ++i) {
    for(int d = 0; d < (numDofs - 1); ++d) {
      outArr[(numDofs * i) + d] = solArr[((numDofs - 1) * i) + d];
    }//end d
    uArr[i] = solArr[((numDofs - 1) * i) + (numDofs - 2)];
  }//end i
  VecRestoreArray((data->sol), &solArr);
  VecRestoreArray((data->u), &uArr);

  applyFD1D(comm, *(data->partX), (data->u), (data->uPrime));

  double* uPrimeArr;
  VecGetArray((data->uPrime), &uPrimeArr);
  for(int i = 0; i < nx; ++i) {
    outArr[(numDofs * i) + (numDofs - 1)] = uPrimeArr[i];
  }//end i
  VecRestoreArray((data->uPrime), &uPrimeArr);

  VecRestoreArray(out, &outArr);

  return 0;
}

void applyFD1D(MPI_Comm comm, std::vector<PetscInt>& partX, Vec in, Vec out) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  int nx = partX[rank];

  double* inArr;
  double* outArr;
  VecGetArray(in, &inArr);
  VecGetArray(out, &outArr);

  /*
  //First Order
  for(int i = 0; i < (nx - 1); ++i) {
  outArr[i] = (inArr[i + 1] - inArr[i])/2.0;
  }//end i
  outArr[nx - 1] = (inArr[nx - 1] - inArr[nx - 2])/2.0;
  */

  //Second Order
  outArr[0] = -((3.0 * inArr[0]) - (4.0 * inArr[1]) + inArr[2])/4.0;
  for(int i = 1; i < (nx - 1); ++i) {
    outArr[i] = (inArr[i + 1] - inArr[i - 1])/4.0;
  }//end i
  outArr[nx - 1] = ((3.0 * inArr[nx - 1]) - (4.0 * inArr[nx - 2]) + inArr[nx - 3])/4.0;

  /*
  //Fourth Order
  outArr[0] = -((25.0 * inArr[0]) - (48.0 * inArr[1]) + (36.0 * inArr[2]) - (16.0 * inArr[3]) + (3.0 * inArr[4]))/24.0;
  outArr[1] = -((25.0 * inArr[1]) - (48.0 * inArr[2]) + (36.0 * inArr[3]) - (16.0 * inArr[4]) + (3.0 * inArr[5]))/24.0;
  for(int i = 2; i < (nx - 2); ++i) {
  outArr[i] = (-inArr[i + 2] + (8.0 * inArr[i + 1]) - (8.0 * inArr[i - 1]) + inArr[i - 2])/24.0;
  }//end i
  outArr[nx - 2] = ((25.0 * inArr[nx - 2]) - (48.0 * inArr[nx - 3]) + (36.0 * inArr[nx - 4]) - (16.0 * inArr[nx - 5]) + (3.0 * inArr[nx - 6]))/24.0;
  outArr[nx - 1] = ((25.0 * inArr[nx - 1]) - (48.0 * inArr[nx - 2]) + (36.0 * inArr[nx - 3]) - (16.0 * inArr[nx - 4]) + (3.0 * inArr[nx - 5]))/24.0;
  */

  VecRestoreArray(in, &inArr);
  VecRestoreArray(out, &outArr);
}

/*
   void applyFD1D(MPI_Comm comm, std::vector<PetscInt>& partX, Vec in, Vec out) {
   int rank;
   MPI_Comm_rank(comm, &rank);

   int nx = partX[rank];
   int px = partX.size();

   double* inArr;
   VecGetArray(in, &inArr);

   double left1;
   double left2;
   double right1;
   double right2;

   MPI_Request reqR1;
   MPI_Request reqR2;
   MPI_Request reqS1;
   MPI_Request reqS2;

   if(rank < (px - 1)) {
   MPI_Irecv(&right1, 1, MPI_DOUBLE, (rank + 1), 2, comm, &reqR2);
   MPI_Isend(&(inArr[nx - 1]), 1, MPI_DOUBLE, (rank + 1), 1, comm, &reqS1);
   }

   if(rank > 0) {
   MPI_Irecv(&left1, 1, MPI_DOUBLE, (rank - 1), 1, comm, &reqR1);
   MPI_Isend(&(inArr[0]), 1, MPI_DOUBLE, (rank - 1), 2, comm, &reqS2);
   }

   if(rank < (px - 1)) {
   MPI_Status status;
   MPI_Wait(&reqR2, &status);
   }

   MPI_Request reqR3;
   MPI_Request reqS3;

   if(partX[0] == 1) {
   if(rank == 1) {
   if(partX[1] == 1) {
   MPI_Isend(&right1, 1, MPI_DOUBLE, 0, 3, comm, &reqS3);
   } else {
   MPI_Isend(&(in[1]), 1, MPI_DOUBLE, 0, 3, comm, &reqS3);
   }
   }
   if(rank == 0) {
   MPI_Irecv(&right2, 1, MPI_DOUBLE, 1, 3, comm, &reqR3);
   }
   }

   if(rank > 0) {
   MPI_Status status;
   MPI_Wait(&reqR1, &status);
   }

   MPI_Request reqR4;
   MPI_Request reqS4;

   if(partX[px - 1] == 1) {
   if(rank == (px - 2)) {
   if(partX[px - 2] == 1) {
   MPI_Isend(&left1, 1, MPI_DOUBLE, (px - 1), 4, comm, &reqS4);
   } else {
   MPI_Isend(&(in[nx - 2]), 1, MPI_DOUBLE, (px - 1), 4, comm, &reqS4);
   }
   } 
   if(rank == (px - 1)) {
   MPI_Irecv(&left2, 1, MPI_DOUBLE, (px - 2), 4, comm, &reqR4);
   }
}

if(rank < (px - 1)) {
  MPI_Status status;
  MPI_Wait(&reqS1, &status);
}

if(rank > 0) {
  MPI_Status status;
  MPI_Wait(&reqS2, &status);
}

if(partX[0] == 1) {
  if(rank == 1) {
    MPI_Status status;
    MPI_Wait(&reqS3, &status);
  }
  if(rank == 0) {
    MPI_Status status;
    MPI_Wait(&reqR3, &status);
  }
}

if(partX[px - 1] == 1) {
  if(rank == (px - 2)) {
    MPI_Status status;
    MPI_Wait(&reqS4, &status);
  } 
  if(rank == (px - 1)) {
    MPI_Status status;
    MPI_Wait(&reqR4, &status);
  }
}

double* outArr;
VecGetArray(out, &outArr);

for(int i = 1; i < (nx - 1); ++i) {
  outArr[i] = 0.25*(inArr[i + 1] - inArr[i - 1]);
}//end i

if(rank == 0) {
  if(nx == 1) {
    outArr[0] = -0.25*((3.0*inArr[0]) - (4.0*right1) + right2);
  } else if(nx == 2) {
    outArr[0] = -0.25*((3.0*inArr[0]) - (4.0*inArr[1]) + right1);
    outArr[1] = 0.25*(right1 - inArr[0]);
  } else {
    outArr[0] = -0.25*((3.0*inArr[0]) - (4.0*inArr[1]) + inArr[2]);
    if(px == 1) {
      outArr[nx - 1] = 0.25*((3.0*inArr[nx - 1]) - (4.0*inArr[nx - 2]) + inArr[nx - 3]);
    } else {
      outArr[nx - 1] = 0.25*(right1 - inArr[nx - 2]);
    }
  }
} else if(rank == (px - 1)) {
  if(nx == 1) {
    outArr[0] = 0.25*((3.0*inArr[0]) - (4.0*left1) + left2);
  } else if(nx == 2) {
    outArr[0] = 0.25*(inArr[1] - left1);
    outArr[1] = 0.25*((3.0*inArr[1]) - (4.0*inArr[0]) + left1);
  } else {
    outArr[0] = 0.25*(inArr[1] - left1);
    outArr[nx - 1] = 0.25*((3.0*inArr[nx - 1]) - (4.0*inArr[nx - 2]) + inArr[nx - 3]);
  }
} else {
  if(nx == 1) {
    outArr[0] = 0.25*(right1 - left1);
  } else {
    outArr[0] = 0.25*(inArr[1] - left1);
    outArr[nx - 1] = 0.25*(right1 - inArr[nx - 2]);
  }
}

VecRestoreArray(in, &inArr);
VecRestoreArray(out, &outArr);
}
*/

void destroyKSP(std::vector<KSP>& ksp) {
  for(size_t i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(&(ksp[i]));
    }
  }//end i
  ksp.clear();
}

void destroyPCFD1Ddata(PCFD1Ddata* data) {
  KSPDestroy(&(data->ksp));
  VecDestroy(&(data->rhs));
  VecDestroy(&(data->sol));
  VecDestroy(&(data->u));
  VecDestroy(&(data->uPrime));
  delete data;
}

void destroyKhat1Ddata(Khat1Ddata* data) {
  VecDestroy(&(data->u));
  VecDestroy(&(data->uPrime));
  VecDestroy(&(data->tmpOut));
  delete data;
}



