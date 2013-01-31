
#include <vector>
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

PetscErrorCode Khat1Dmult(Mat mat, Vec in, Vec out) {
  Khat1Ddata* data;
  MatShellGetContext(mat, &data);

  MPI_Comm comm;
  PetscObjectGetComm(mat, &comm);

  MPI_Comm_rank(comm, &rank);

  int nx = (*(data->partX))[rank];
  int numDofs = data->numDofs;

  double* uArr;
  VecGetArray((data->u), &uArr);

  double* inArr;
  VecGetArray((data->in), &inArr);

  for(int i = 0; i < nx; ++i) {
    uArr[i] = inArr[(numDofs * i) + (numDofs - 1)];
  }//end i

  VecRestoreArray((data->u), &uArr);
  VecRestoreArray((data->in), &inArr);

  applyFD1D(comm, *(data->partX), (data->u), (data->uPrime));
  MatMult((data->K12), (data->uPrime), (data->tmpOut));
  MatMult((data->K11), in, out);
  VecAXPY(out, 1.0, (data->tmpOut));

  return 0;
}

PetscErrorCode Kcol1Dmult(Mat mat, Vec in, Vec out) {
  Kcol1Ddata* data;
  MatShellGetContext(mat, &data);

  MatMult((data->Kl), in, (data->l));
  MatMult((data->Kh), in, (data->h));

  double* outArr;
  VecGetArray(out, &outArr);

  double* lArr;
  VecGetArray((data->l), &lArr);

  double* hArr;
  VecGetArray((data->h), &hArr);

  int nx = data->nx;
  int numDofs = data->numDofs;

  for(int i = 0; i < nx; ++i) {
    for(int d = 0; d < (numDofs - 1); ++d) {
      outArr[(numDofs * i) + d] = lArr[((numDofs - 1) * i) + d];
    }//end d
    outArr[(numDofs * i) + (numDofs - 1)] = hArr[i];
  }//end i

  VecRestoreArray((data->l), &lArr);
  VecRestoreArray((data->h), &hArr);
  VecRestoreArray(out, &outArr);

  return 0;
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

void destroyKSP(std::vector<KSP>& ksp) {
  for(size_t i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(&(ksp[i]));
    }
  }//end i
  ksp.clear();
}


