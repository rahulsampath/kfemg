
#include <vector>
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

extern PetscLogEvent vCycleEvent;

PetscErrorCode applyMG(PC pc, Vec in, Vec out) {
  MGdata* data;
  PCShellGetContext(pc, (void**)(&data));

  VecZeroEntries(out);
  data->mgSol[data->Kmat.size() - 1] = out;
  data->mgRhs[data->Kmat.size() - 1] = in;
  for(int iter = 0; iter < data->numVcycles; ++iter) {
    applyVcycle((data->Kmat.size() - 1), data->Kmat, data->Pmat, data->tmpCvec,
        data->ksp, data->mgSol, data->mgRhs, data->mgRes);
  }//end iter
  data->mgSol[data->Kmat.size() - 1] = NULL;
  data->mgRhs[data->Kmat.size() - 1] = NULL;

  return 0;
}

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<KSP>& ksp, std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes) {
  PetscLogEventBegin(vCycleEvent, 0, 0, 0, 0);
#ifdef DEBUG
  assert(ksp[currLev] != NULL);
#endif
  KSPSolve(ksp[currLev], mgRhs[currLev], mgSol[currLev]);
  if(currLev > 0) {
    computeResidual(Kmat[currLev], mgSol[currLev], mgRhs[currLev], mgRes[currLev]);
    applyRestriction(Pmat[currLev - 1], tmpCvec[currLev - 1], mgRes[currLev], mgRhs[currLev - 1]);
    if(ksp[currLev - 1] != NULL) {
      if(currLev > 1) {
        VecZeroEntries(mgSol[currLev - 1]);
      }
      applyVcycle((currLev - 1), Kmat, Pmat, tmpCvec, ksp, mgSol, mgRhs, mgRes);
    }
    applyProlongation(Pmat[currLev - 1], tmpCvec[currLev - 1], mgSol[currLev - 1], mgRes[currLev]);
    VecAXPY(mgSol[currLev], 1.0, mgRes[currLev]);
    KSPSolve(ksp[currLev], mgRhs[currLev], mgSol[currLev]);
  }
  PetscLogEventEnd(vCycleEvent, 0, 0, 0, 0);
}

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
    std::vector<PCShellData>& data, int dim, int dofsPerNode, bool print) {
  ksp.resize((Kmat.size()), NULL);
  for(int lev = 0; lev < (Kmat.size()); ++lev) {
    if(Kmat[lev] != NULL) {
      PC pc;
      KSPCreate(activeComms[lev], &(ksp[lev]));
      KSPGetPC(ksp[lev], &pc);
      if(lev == 0) {
        KSPSetType(ksp[lev], KSPPREONLY);
        KSPSetInitialGuessNonzero(ksp[lev], PETSC_FALSE);
        PCSetType(pc, PCLU);
        KSPSetOptionsPrefix(ksp[lev], "coarse_");
      } else {
        KSPSetType(ksp[lev], KSPFGMRES);
        KSPSetPCSide(ksp[lev], PC_RIGHT);
        PCSetType(pc, PCSHELL);
        PCShellSetContext(pc, &(data[lev]));
        PCShellSetApply(pc, &applyShellPC);
        KSPSetInitialGuessNonzero(ksp[lev], PETSC_TRUE);
        KSPSetOptionsPrefix(ksp[lev], "smooth_");
      }
      KSPSetOperators(ksp[lev], Kmat[lev], Kmat[lev], SAME_PRECONDITIONER);
      KSPSetTolerances(ksp[lev], 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetFromOptions(ksp[lev]);
    }
  }//end lev
}

void createPCShellData(std::vector<PCShellData>& data, std::vector<std::vector<Mat> >& KblkDiag,
    std::vector<std::vector<Mat> >& KblkUpper, bool print) {
  PetscInt numBlkIters = 2;
  PetscOptionsGetInt(PETSC_NULL, "-numBlkIters", &numBlkIters, PETSC_NULL);
  if(print) {
    std::cout<<"numBlkIters = "<<numBlkIters<<std::endl;
  }
  PetscBool allBlocksSame = PETSC_TRUE;
  PetscOptionsGetBool(PETSC_NULL, "-allBlocksSame", &allBlocksSame, PETSC_NULL);
  if(print) {
    std::cout<<"allBlocksSame = "<<allBlocksSame<<std::endl; 
  }
  data.resize(KblkDiag.size());
  for(int i = 0; i < data.size(); ++i) {
    data[i].numBlkIters = numBlkIters; 
    data[i].KblkDiag = KblkDiag[i];
    data[i].KblkUpper = KblkUpper[i];
    data[i].blkKsp.resize(KblkDiag[i].size(), NULL);
    for(int j = 0; j < KblkDiag[i].size(); ++j) {
      MPI_Comm comm;
      PC pc;
      KSP ksp;
      PetscObjectGetComm(((PetscObject)(KblkDiag[i][j])), &comm);
      KSPCreate(comm, &ksp);
      KSPGetPC(ksp, &pc);
      KSPSetType(ksp, KSPCG);
      KSPSetPCSide(ksp, PC_LEFT);
      PCSetType(pc, PCJACOBI);
      KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
      KSPSetOperators(ksp, KblkDiag[i][j], KblkDiag[i][j], SAME_PRECONDITIONER);
      KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      char str[256];
      if(allBlocksSame) {
        sprintf(str, "block_");
      } else {
        sprintf(str, "block%d_", j);
      }
      KSPSetOptionsPrefix(ksp, str);
      KSPSetFromOptions(ksp);
      data[i].blkKsp[j] = ksp;
    }//end j
    if(!(KblkDiag[i].empty())) {
      MatGetVecs(KblkDiag[i][0], &(data[i].diagIn), &(data[i].diagOut));
    } else {
      data[i].diagIn = NULL;
      data[i].diagOut = NULL;
    }
    data[i].upperIn.resize(KblkUpper[i].size(), NULL);
    for(int j = 0; j < KblkUpper[i].size(); ++j) {
      MatGetVecs(KblkUpper[i][j], &(data[i].upperIn[j]), NULL);
    }//end j
  }//end i
}

PetscErrorCode applyShellPC(PC pc, Vec in, Vec out) {
  PCShellData* data;
  PCShellGetContext(pc, (void**)(&data));

  unsigned int dofsPerNode = data->KblkDiag.size();

  PetscInt locSz;
  VecGetLocalSize(in, &locSz);

  int blkSz = locSz/dofsPerNode;

  VecZeroEntries(out);

  for(int iter = 0; iter < (data->numBlkIters); ++iter) {
    PetscScalar* arr1;
    PetscScalar* arr2;
    if((iter > 0) && (dofsPerNode > 1)) {
      VecGetArray(out, &arr1);
      VecGetArray(data->upperIn[0], &arr2);
      for(int i = 0; i < blkSz; ++i) {
        for(int d = 0; d < (dofsPerNode - 1); ++d) {
          arr2[(i*(dofsPerNode - 1)) + d] = arr1[(i*dofsPerNode) + 1 + d]; 
        }//end d
      }//end i
      VecRestoreArray(data->upperIn[0], &arr2);
      VecRestoreArray(out, &arr1);

      MatMult(data->KblkUpper[0], data->upperIn[0], data->diagOut);

      VecGetArray(in, &arr1);
      VecGetArray(data->diagOut, &arr2);
      for(int i = 0; i < blkSz; ++i) {
        arr2[i] = arr1[i*dofsPerNode] - arr2[i];
      }//end i
      VecRestoreArray(data->diagOut, &arr2);
      VecRestoreArray(in, &arr1);
    } else {
      VecGetArray(in, &arr1);
      VecGetArray(data->diagOut, &arr2);
      for(int i = 0; i < blkSz; ++i) {
        arr2[i] = arr1[i*dofsPerNode];
      }//end i
      VecRestoreArray(data->diagOut, &arr2);
      VecRestoreArray(in, &arr1);
    }

    VecGetArray(out, &arr1);
    VecGetArray(data->diagIn, &arr2);
    for(int i = 0; i < blkSz; ++i) {
      arr2[i] = arr1[i*dofsPerNode];
    }//end i
    VecRestoreArray(data->diagIn, &arr2);
    VecRestoreArray(out, &arr1);

    KSPSolve(data->blkKsp[0], data->diagOut, data->diagIn);

    VecGetArray(out, &arr1);
    VecGetArray(data->diagIn, &arr2);
    for(int i = 0; i < blkSz; ++i) {
      arr1[i*dofsPerNode] = arr2[i];
    }//end i
    VecRestoreArray(data->diagIn, &arr2);
    VecRestoreArray(out, &arr1);

    if(dofsPerNode > 1) {
      MatMultTranspose(data->KblkUpper[0], data->diagIn, data->upperIn[0]);

      VecGetArray(in, &arr1);
      VecGetArray(data->upperIn[0], &arr2);
      for(int i = 0; i < blkSz; ++i) {
        for(int d = 0; d < (dofsPerNode - 1); ++d) {
          arr2[(i*(dofsPerNode - 1)) + d] = arr1[(i*dofsPerNode) + 1 + d] - arr2[(i*(dofsPerNode - 1)) + d];
        }//end d
      }//end i
      VecRestoreArray(data->upperIn[0], &arr2);
      VecRestoreArray(in, &arr1);
    }

    for(int dof = 1; dof < dofsPerNode; ++dof) {
      if((iter > 0) && (dofsPerNode > (dof + 1))) {
        VecGetArray(out, &arr1);
        VecGetArray(data->upperIn[dof], &arr2);
        for(int i = 0; i < blkSz; ++i) {
          for(int d = 0; d < (dofsPerNode - 1 - dof); ++d) {
            arr2[(i*(dofsPerNode - 1 - dof)) + d] = arr1[(i*dofsPerNode) + 1 + dof + d];
          }//end d
        }//end i
        VecRestoreArray(data->upperIn[dof], &arr2);
        VecRestoreArray(out, &arr1);

        MatMult(data->KblkUpper[dof], data->upperIn[dof], data->diagOut);

        VecGetArray(data->upperIn[dof - 1], &arr1);
        VecGetArray(data->diagOut, &arr2);
        for(int i = 0; i < blkSz; ++i) {
          arr2[i] = arr1[i*(dofsPerNode - dof)] - arr2[i];
        }//end i
        VecRestoreArray(data->diagOut, &arr2);
        VecRestoreArray(data->upperIn[dof - 1], &arr1);
      } else {
        VecGetArray(data->upperIn[dof - 1], &arr1);
        VecGetArray(data->diagOut, &arr2);
        for(int i = 0; i < blkSz; ++i) {
          arr2[i] = arr1[i*(dofsPerNode - dof)];
        }//end i
        VecRestoreArray(data->diagOut, &arr2);
        VecRestoreArray(data->upperIn[dof - 1], &arr1);
      }

      VecGetArray(out, &arr1);
      VecGetArray(data->diagIn, &arr2);
      for(int i = 0; i < blkSz; ++i) {
        arr2[i] = arr1[(i*dofsPerNode) + dof];
      }//end i
      VecRestoreArray(data->diagIn, &arr2);
      VecRestoreArray(out, &arr1);

      KSPSolve(data->blkKsp[dof], data->diagOut, data->diagIn);

      VecGetArray(out, &arr1);
      VecGetArray(data->diagIn, &arr2);
      for(int i = 0; i < blkSz; ++i) {
        arr1[(i*dofsPerNode) + dof] = arr2[i];
      }//end i
      VecRestoreArray(data->diagIn, &arr2);
      VecRestoreArray(out, &arr1);

      if(dofsPerNode > (dof + 1)) {
        MatMultTranspose(data->KblkUpper[dof], data->diagIn, data->upperIn[dof]);

        VecGetArray(data->upperIn[dof - 1], &arr1);
        VecGetArray(data->upperIn[dof], &arr2);
        for(int i = 0; i < blkSz; ++i) {
          for(int d = 0; d < (dofsPerNode - 1 - dof); ++d) {
            arr2[(i*(dofsPerNode - 1 - dof)) + d] = arr1[(i*(dofsPerNode - dof)) + 1 + d] - arr2[(i*(dofsPerNode - 1 - dof)) + d];
          }//end d
        }//end i
        VecRestoreArray(data->upperIn[dof], &arr2);
        VecRestoreArray(data->upperIn[dof - 1], &arr1);
      }
    }//end dof
  }//end iter

  return 0;
}

void destroyPCShellData(std::vector<PCShellData>& data) {
  for(int i = 0; i < data.size(); ++i) {
    data[i].KblkDiag.clear();
    data[i].KblkUpper.clear();
    destroyKSP(data[i].blkKsp);
    if((data[i].diagIn) != NULL) {
      VecDestroy(&(data[i].diagIn));
    }
    if((data[i].diagOut) != NULL) {
      VecDestroy(&(data[i].diagOut));
    }
    destroyVec(data[i].upperIn);
  }//end i
}

void destroyKSP(std::vector<KSP>& ksp) {
  for(int i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(&(ksp[i]));
    }
  }//end i
  ksp.clear();
}



