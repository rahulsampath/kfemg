
#include <vector>
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

extern PetscLogEvent vCycleEvent;

void createAllSchurPC(std::vector<std::vector<SchurPCdata> >& pcData, std::vector<std::vector<Mat> >& SmatShells,
    std::vector<Mat>& KmatShells, std::vector<std::vector<KmatData> >& kMatData,
    std::vector<std::vector<SmatData> >& sMatData) {
  pcData.resize(SmatShells.size());
  for(size_t lev = 0; lev < pcData.size(); ++lev) {
    createSchurPCdata(lev, pcData[lev], SmatShells[lev], KmatShells[lev], kMatData[lev], sMatData[lev]);
  }//end lev
}

void createSchurPCdata(int lev, std::vector<SchurPCdata>& pcData, std::vector<Mat>& SmatShell, Mat& KmatShell, 
    std::vector<KmatData>& kMatData, std::vector<SmatData>& sMatData) {
  pcData.resize(SmatShell.size());
  for(size_t i = 0; i < pcData.size(); ++i) {
    pcData[i].B = (sMatData[i].B);
    MatGetVecs(SmatShell[i], &(pcData[i].sSol), &(pcData[i].sRhs));
    MatGetVecs((pcData[i].B), &(pcData[i].x), NULL);
    VecDuplicate((pcData[i].x), &(pcData[i].z));
    VecDuplicate((pcData[i].x), &(pcData[i].cRhs));
    MPI_Comm comm;
    PetscObjectGetComm(((PetscObject)(pcData[i].B)), &comm);
    PC spc;
    PC cpc;
    PC sCpc;
    KSPCreate(comm, &(pcData[i].sKsp));
    KSPCreate(comm, &(pcData[i].cKsp));
    KSPCreate(comm, &(sMatData[i].cKsp));
    KSPGetPC((pcData[i].sKsp), &spc);
    KSPGetPC((pcData[i].cKsp), &cpc);
    KSPGetPC((sMatData[i].cKsp), &sCpc);
    if(lev == 0) {
      PCSetType(spc, PCCHOLESKY);
      KSPSetOptionsPrefix((pcData[i].sKsp), "coarseSchurS_");
    } else {
      PCSetType(spc, PCJACOBI);
      KSPSetOptionsPrefix((pcData[i].sKsp), "smoothSchurS_");
    }
    KSPSetType((pcData[i].sKsp), KSPFGMRES);
    KSPSetPCSide((pcData[i].sKsp), PC_RIGHT);
    if((i + 1) == (pcData.size())) {
      if(lev == 0) {
        PCSetType(cpc, PCCHOLESKY);
        PCSetType(sCpc, PCCHOLESKY);
        KSPSetOptionsPrefix((pcData[i].cKsp), "coarseSchurLastC_");
        KSPSetOptionsPrefix((sMatData[i].cKsp), "coarseSmatLastC_");
      } else {
        PCSetType(cpc, PCJACOBI);
        PCSetType(sCpc, PCJACOBI);
        KSPSetOptionsPrefix((pcData[i].cKsp), "smoothSchurLastC_");
        KSPSetOptionsPrefix((sMatData[i].cKsp), "smoothSmatLastC_");
      }
      KSPSetType((pcData[i].cKsp), KSPCG);
      KSPSetType((sMatData[i].cKsp), KSPCG);
      KSPSetPCSide((pcData[i].cKsp), PC_LEFT);
      KSPSetPCSide((sMatData[i].cKsp), PC_LEFT);
    } else {
      if(lev == 0) {
        KSPSetOptionsPrefix((pcData[i].cKsp), "coarseSchurC_");
        KSPSetOptionsPrefix((sMatData[i].cKsp), "coarseSmatC_");
      } else {
        KSPSetOptionsPrefix((pcData[i].cKsp), "smoothSchurC_");
        KSPSetOptionsPrefix((sMatData[i].cKsp), "smoothSmatC_");
      }
      PCSetType(cpc, PCSHELL);
      PCSetType(sCpc, PCSHELL);
      PCShellSetContext(cpc, &(pcData[i + 1]));
      PCShellSetContext(sCpc, &(pcData[i + 1]));
      PCShellSetApply(cpc, &applySchurPC);
      PCShellSetApply(sCpc, &applySchurPC);
      PCShellSetName(cpc, "MySchurPC");
      PCShellSetName(sCpc, "MySchurPC");
      KSPSetType((pcData[i].cKsp), KSPFGMRES);
      KSPSetType((sMatData[i].cKsp), KSPFGMRES);
      KSPSetPCSide((pcData[i].cKsp), PC_RIGHT);
      KSPSetPCSide((sMatData[i].cKsp), PC_RIGHT);
    }
    KSPSetOperators((pcData[i].sKsp), SmatShell[i], (sMatData[i].A), SAME_PRECONDITIONER);
    if(i == 0) {
      KSPSetOperators((pcData[i].cKsp), KmatShell, KmatShell, SAME_PRECONDITIONER);
      KSPSetOperators((sMatData[i].cKsp), KmatShell, KmatShell, SAME_PRECONDITIONER);
    } else {
      KSPSetOperators((pcData[i].cKsp), (kMatData[i - 1].C), (kMatData[i - 1].C), SAME_PRECONDITIONER);
      KSPSetOperators((sMatData[i].cKsp), (kMatData[i - 1].C), (kMatData[i - 1].C), SAME_PRECONDITIONER);
    }
    KSPSetInitialGuessNonzero((pcData[i].sKsp), PETSC_FALSE);
    KSPSetInitialGuessNonzero((pcData[i].cKsp), PETSC_FALSE);
    KSPSetInitialGuessNonzero((sMatData[i].cKsp), PETSC_FALSE);
    KSPSetTolerances((pcData[i].sKsp), 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
    KSPSetTolerances((pcData[i].cKsp), 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
    KSPSetTolerances((sMatData[i].cKsp), 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
    KSPSetFromOptions((pcData[i].sKsp));
    KSPSetFromOptions((pcData[i].cKsp));
    KSPSetFromOptions((sMatData[i].cKsp));
  }//end i
}

void createAllSmatShells(std::vector<std::vector<Mat> >& SmatShells, std::vector<std::vector<SmatData> >& sMatData,
    std::vector<std::vector<Mat> >& KblkDiag, std::vector<std::vector<Mat> >& KblkUpper) {
  SmatShells.resize(KblkDiag.size());
  sMatData.resize(SmatShells.size());
  for(size_t i = 0; i < SmatShells.size(); ++i) {
    createSmatData(SmatShells[i], sMatData[i], KblkDiag[i], KblkUpper[i]);
  }//end i
}

void createSmatData(std::vector<Mat>& SmatShell, std::vector<SmatData>& data,
    std::vector<Mat>& KblkDiag, std::vector<Mat>& KblkUpper) {
  SmatShell.resize((KblkUpper.size()), NULL);
  data.resize(SmatShell.size());
  for(size_t i = 0; i < SmatShell.size(); ++i) {
    MPI_Comm comm;
    PetscObjectGetComm(((PetscObject)(KblkDiag[i])), &comm);
    PetscInt locSz;
    MatGetLocalSize((KblkDiag[i]), PETSC_NULL, &locSz);
    MatCreateShell(comm, locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE, &(data[i]), &(SmatShell[i]));
    MatShellSetOperation(SmatShell[i], MATOP_MULT, ((void(*)(void))(&applySmatvec)));
    data[i].A = KblkDiag[i];
    data[i].B = KblkUpper[i];
    MatGetVecs((data[i].B), &(data[i].cSol), &(data[i].aOut));
    VecDuplicate((data[i].cSol), &(data[i].cRhs));
  }//end i
}

void createAllKmatShells(std::vector<Mat>& KmatShells, std::vector<std::vector<KmatData> >& kMatData,
    std::vector<std::vector<Mat> >& KblkDiag, std::vector<std::vector<Mat> >& KblkUpper) {
  KmatShells.resize(KblkDiag.size(), NULL);
  kMatData.resize(KmatShells.size());
  for(size_t i = 0; i < KmatShells.size(); ++i) {
    createKmatData(1, KmatShells[i], kMatData[i], KblkDiag[i], KblkUpper[i]);
  }//end i
}

void createKmatData(int blkId, Mat& KmatShell, std::vector<KmatData>& data,
    std::vector<Mat>& KblkDiag, std::vector<Mat>& KblkUpper) {
#ifdef DEBUG
  assert(blkId > 0);
#endif
  if(blkId < (KblkDiag.size())) {
    if(blkId == 1) {
      data.resize((KblkDiag.size()) - 2);
    }
    if((blkId + 1) == (KblkDiag.size())) {
      KmatShell = KblkDiag[blkId];
    } else {
      MPI_Comm comm;
      PetscObjectGetComm(((PetscObject)(KblkDiag[blkId])), &comm);
      PetscInt locDiagSz;
      MatGetLocalSize((KblkDiag[blkId]), PETSC_NULL, &locDiagSz);
      PetscInt locUpperSz;
      MatGetLocalSize((KblkUpper[blkId]), PETSC_NULL, &locUpperSz);
      PetscInt locSz = locDiagSz + locUpperSz;
      MatCreateShell(comm, locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE, &(data[blkId - 1]), &KmatShell);
      MatShellSetOperation(KmatShell, MATOP_MULT, ((void(*)(void))(&applyKmatvec)));
      data[blkId - 1].A = KblkDiag[blkId];
      data[blkId - 1].B = KblkUpper[blkId]; 
      createKmatData((blkId + 1), (data[blkId - 1].C), data, KblkDiag, KblkUpper); 
      MatGetVecs((data[blkId - 1].A), &(data[blkId - 1].aIn), &(data[blkId - 1].aOut));
      MatGetVecs((data[blkId - 1].B), &(data[blkId - 1].bIn), NULL);
      MatGetVecs((data[blkId - 1].C), NULL, &(data[blkId - 1].cOut));
    }
  } else {
    KmatShell = NULL;
  }
}

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
      VecZeroEntries(mgSol[currLev - 1]);
      applyVcycle((currLev - 1), Kmat, Pmat, tmpCvec, ksp, mgSol, mgRhs, mgRes);
    }
    applyProlongation(Pmat[currLev - 1], tmpCvec[currLev - 1], mgSol[currLev - 1], mgRes[currLev]);
    VecAXPY(mgSol[currLev], 1.0, mgRes[currLev]);
    KSPSolve(ksp[currLev], mgRhs[currLev], mgSol[currLev]);
  }
  PetscLogEventEnd(vCycleEvent, 0, 0, 0, 0);
}

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
    std::vector<std::vector<SchurPCdata> >& data, int dim, int dofsPerNode, bool print) {
  ksp.resize((Kmat.size()), NULL);
  for(int lev = 0; lev < (Kmat.size()); ++lev) {
    if(Kmat[lev] != NULL) {
      PC pc;
      KSPCreate(activeComms[lev], &(ksp[lev]));
      KSPGetPC(ksp[lev], &pc);
      if(lev == 0) {
        KSPSetOptionsPrefix(ksp[lev], "coarse_");
      } else {
        KSPSetOptionsPrefix(ksp[lev], "smooth_");
      }
      if(dofsPerNode == 1) {
        KSPSetType(ksp[lev], KSPCG);
        KSPSetPCSide(ksp[lev], PC_LEFT);
        if(lev == 0) {
          PCSetType(pc, PCCHOLESKY);
        } else {
          PCSetType(pc, PCJACOBI);
        }
      } else {
        KSPSetType(ksp[lev], KSPFGMRES);
        KSPSetPCSide(ksp[lev], PC_RIGHT);
        PCSetType(pc, PCSHELL);
        PCShellSetContext(pc, &(data[lev][0]));
        PCShellSetApply(pc, &applySchurPC);
        PCShellSetName(pc, "MySchurPC");
      }
      KSPSetInitialGuessNonzero(ksp[lev], PETSC_TRUE);
      KSPSetOperators(ksp[lev], Kmat[lev], Kmat[lev], SAME_PRECONDITIONER);
      KSPSetTolerances(ksp[lev], 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetFromOptions(ksp[lev]);
    }
  }//end lev
}

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
    std::vector<BlockPCdata>& data, int dim, int dofsPerNode, bool print) {
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
        PCShellSetApply(pc, &applyBlockPC);
        PCShellSetName(pc, "MyBlockPC");
        KSPSetInitialGuessNonzero(ksp[lev], PETSC_TRUE);
        KSPSetOptionsPrefix(ksp[lev], "smooth_");
      }
      KSPSetOperators(ksp[lev], Kmat[lev], Kmat[lev], SAME_PRECONDITIONER);
      KSPSetTolerances(ksp[lev], 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetFromOptions(ksp[lev]);
    }
  }//end lev
}

void createBlockPCdata(std::vector<BlockPCdata>& data, std::vector<std::vector<Mat> >& KblkDiag,
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
  for(size_t i = 0; i < data.size(); ++i) {
    data[i].numBlkIters = numBlkIters; 
    data[i].KblkDiag = KblkDiag[i];
    data[i].KblkUpper = KblkUpper[i];
    data[i].blkKsp.resize(KblkDiag[i].size(), NULL);
    for(unsigned int j = 0; j < KblkDiag[i].size(); ++j) {
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
        sprintf(str, "block%u_", j);
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
    for(size_t j = 0; j < KblkUpper[i].size(); ++j) {
      MatGetVecs(KblkUpper[i][j], &(data[i].upperIn[j]), NULL);
    }//end j
  }//end i
}

PetscErrorCode applyBlockPC(PC pc, Vec in, Vec out) {
  BlockPCdata* data;
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

    for(unsigned int dof = 1; dof < dofsPerNode; ++dof) {
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

PetscErrorCode applySchurPC(PC pc, Vec in, Vec out) {
  SchurPCdata* data;
  PCShellGetContext(pc, (void**)(&data));

  PetscInt subLocSz;
  PetscInt fullLocSz;
  VecGetLocalSize(data->sSol, &subLocSz);
  VecGetLocalSize(in, &fullLocSz);
  PetscInt numDofs = fullLocSz/subLocSz;

  PetscScalar* inArr;
  PetscScalar* cRhsArr;
  PetscScalar* sRhsArr;
  VecGetArray(in, &inArr);
  VecGetArray(data->cRhs, &cRhsArr);
  VecGetArray(data->sRhs, &sRhsArr);
  for(PetscInt i = 0; i < subLocSz; ++i) {
    PetscInt inBase = i*numDofs;
    sRhsArr[i] = inArr[inBase];
    PetscInt cRhsBase = i*(numDofs - 1);
    for(PetscInt d = 1; d < numDofs; ++d) {
      cRhsArr[cRhsBase + d - 1] = inArr[inBase + d];
    }//end d
  }//end i
  VecRestoreArray(data->sRhs, &sRhsArr);
  VecRestoreArray(data->cRhs, &cRhsArr);
  VecRestoreArray(in, &inArr);

  KSPSolve(data->cKsp, data->cRhs, data->x);
  VecScale(data->x, -1.0);
  MatMultAdd(data->B, data->x, data->sRhs, data->sRhs);
  KSPSolve(data->sKsp, data->sRhs, data->sSol);
  MatMultTranspose(data->B, data->sSol, data->cRhs);
  KSPSolve(data->cKsp, data->cRhs, data->z);
  VecAXPBY(data->x, -1.0, -1.0, data->z);

  PetscScalar* outArr;
  PetscScalar* xArr;
  PetscScalar* sSolArr;
  VecGetArray(out, &outArr);
  VecGetArray(data->x, &xArr);
  VecGetArray(data->sSol, &sSolArr);
  for(PetscInt i = 0; i < subLocSz; ++i) {
    PetscInt outBase = i*numDofs;
    outArr[outBase] = sSolArr[i];
    PetscInt xBase = i*(numDofs - 1);
    for(PetscInt d = 1; d < numDofs; ++d) {
      outArr[outBase + d] = xArr[xBase + d - 1];
    }//end d
  }//end i
  VecRestoreArray(data->sSol, &sSolArr);
  VecRestoreArray(data->x, &xArr);
  VecRestoreArray(out, &outArr);

  return 0;
}

PetscErrorCode applyKmatvec(Mat Kmat, Vec in, Vec out) {
  KmatData* data;
  MatShellGetContext(Kmat, &data);

  PetscInt subLocSz;
  PetscInt fullLocSz;
  VecGetLocalSize(data->aIn, &subLocSz);
  VecGetLocalSize(in, &fullLocSz);
  PetscInt numDofs = fullLocSz/subLocSz;

  PetscScalar* inArr;
  PetscScalar* aInArr;
  PetscScalar* bInArr;
  VecGetArray(in, &inArr);
  VecGetArray(data->aIn, &aInArr);
  VecGetArray(data->bIn, &bInArr);
  for(PetscInt i = 0; i < subLocSz; ++i) {
    PetscInt inBase = i*numDofs;
    aInArr[i] = inArr[inBase];
    PetscInt bInBase = i*(numDofs - 1);
    for(PetscInt d = 1; d < numDofs; ++d) {
      bInArr[bInBase + d - 1] = inArr[inBase + d];
    }//end d
  }//end i
  VecRestoreArray(data->bIn, &bInArr);
  VecRestoreArray(data->aIn, &aInArr);
  VecRestoreArray(in, &inArr);

  MatMult(data->A, data->aIn, data->aOut);
  MatMultAdd(data->B, data->bIn, data->aOut, data->aOut);
  MatMult(data->C, data->bIn, data->cOut);
  MatMultTransposeAdd(data->B, data->aIn, data->cOut, data->cOut);

  PetscScalar* outArr;
  PetscScalar* aOutArr;
  PetscScalar* cOutArr;
  VecGetArray(out, &outArr);
  VecGetArray(data->aOut, &aOutArr);
  VecGetArray(data->cOut, &cOutArr);
  for(PetscInt i = 0; i < subLocSz; ++i) {
    PetscInt outBase = i*numDofs;
    outArr[outBase] = aOutArr[i];
    PetscInt cOutBase = i*(numDofs - 1);
    for(PetscInt d = 1; d < numDofs; ++d) {
      outArr[outBase + d] = cOutArr[cOutBase + d - 1];
    }//end d
  }//end i
  VecRestoreArray(data->cOut, &cOutArr);
  VecRestoreArray(data->aOut, &aOutArr);
  VecRestoreArray(out, &outArr);

  return 0;
}

PetscErrorCode applySmatvec(Mat Smat, Vec in, Vec out) {
  SmatData* data;
  MatShellGetContext(Smat, &data);

  MatMult(data->A, in, data->aOut);
  MatMultTranspose(data->B, in, data->cRhs);
  KSPSolve(data->cKsp, data->cRhs, data->cSol);
  VecScale(data->cSol, -1.0);
  MatMultAdd(data->B, data->cSol, data->aOut, out);

  return 0;
}

void destroyBlockPCdata(std::vector<BlockPCdata>& data) {
  for(size_t i = 0; i < data.size(); ++i) {
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
  for(size_t i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(&(ksp[i]));
    }
  }//end i
  ksp.clear();
}

void destroyKmatData(std::vector<KmatData>& data) {
  for(size_t i = 0; i < data.size(); ++i) {
    if((data[i].aIn) != NULL) {
      VecDestroy(&(data[i].aIn));
    }
    if((data[i].aOut) != NULL) {
      VecDestroy(&(data[i].aOut));
    }
    if((data[i].bIn) != NULL) {
      VecDestroy(&(data[i].bIn));
    }
    if((data[i].cOut) != NULL) {
      VecDestroy(&(data[i].cOut));
    }
    if((data[i].C) != NULL) {
      PetscBool same;
      PetscObjectTypeCompare(((PetscObject)(data[i].C)), MATSHELL, &same);
      if(same) {
        MatDestroy(&(data[i].C));
      }
    }
  }//end i
  data.clear();
}

void destroySmatData(std::vector<SmatData>& data) {
  for(size_t i = 0; i < data.size(); ++i) {
    if((data[i].aOut) != NULL) {
      VecDestroy(&(data[i].aOut));
    }
    if((data[i].cRhs) != NULL) {
      VecDestroy(&(data[i].cRhs));
    }
    if((data[i].cSol) != NULL) {
      VecDestroy(&(data[i].cSol));
    }
    if((data[i].cKsp) != NULL) {
      KSPDestroy(&(data[i].cKsp));
    }
  }//end i
  data.clear();
}

void destroySchurPCdata(std::vector<SchurPCdata>& data) {
  for(size_t i = 0; i < data.size(); ++i) {
    if((data[i].cKsp) != NULL) {
      KSPDestroy(&(data[i].cKsp));
    }
    if((data[i].sKsp) != NULL) {
      KSPDestroy(&(data[i].sKsp));
    }
    if((data[i].cRhs) != NULL) {
      VecDestroy(&(data[i].cRhs));
    }
    if((data[i].x) != NULL) {
      VecDestroy(&(data[i].x));
    }
    if((data[i].z) != NULL) {
      VecDestroy(&(data[i].z));
    }
    if((data[i].sRhs) != NULL) {
      VecDestroy(&(data[i].sRhs));
    }
    if((data[i].sSol) != NULL) {
      VecDestroy(&(data[i].sSol));
    }
  }//end i
  data.clear();
}


