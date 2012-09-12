
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscmg.h"
#include "petscksp.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"

int main(int argc, char *argv[]) {
  PetscInitialize(&argc, &argv, "options", PETSC_NULL);
  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);
  std::cout<<"Dim = "<<dim<<std::endl;
  PetscInt K;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);
  std::cout<<"K = "<<K<<std::endl;
  PetscTruth useRandomRHS = PETSC_TRUE;
  PetscOptionsGetTruth(PETSC_NULL, "-useRandomRHS", &useRandomRHS, PETSC_NULL);
  std::cout<<"Random-RHS = "<<useRandomRHS<<std::endl;

  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  int dofsPerNode = getDofsPerNode(dim, K);

  std::vector<PetscInt> Nx;
  std::vector<PetscInt> Ny;
  std::vector<PetscInt> Nz;
  createGridSizes(dim, Nz, Ny, Nx);

  std::vector<DA> da;
  std::vector<MPI_Comm> activeComms;
  std::vector<int> activeNpes;
  createDA(da, activeComms, activeNpes, dofsPerNode, dim, Nz, Ny, Nx, MPI_COMM_WORLD);

  std::vector<Mat> Kmat(da.size(), NULL);
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      DAGetMatrix(da[i], MATAIJ, &(Kmat[i]));
    }
  }//end i

  std::vector<Mat> Pmat((da.size() - 1), NULL);
  for(int lev = 0; lev < (Pmat.size()); ++lev) {
    if(da[lev + 1] != NULL) {
      PetscInt nxf, nyf, nzf;
      DAGetCorners(da[lev + 1], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nxf, &nyf, &nzf);
      MatCreate(activeComms[lev + 1], &(Pmat[lev]));
      PetscInt nxc, nyc, nzc;
      nxc = nyc = nzc = 0;
      if(da[lev] != NULL) {
        DAGetCorners(da[lev], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nxc, &nyc, &nzc);
      }
      if(dim < 3) {
        nzf = nzc = 1;
      }
      if(dim < 2) {
        nyf = nyc = 1;
      }
      PetscInt locRowSz = dofsPerNode*nxf*nyf*nzf;
      PetscInt locColSz = dofsPerNode*nxc*nyc*nzc;
      MatSetSizes(Pmat[lev], locRowSz, locColSz, PETSC_DETERMINE, PETSC_DETERMINE);
      MatSetType(Pmat[lev], MATAIJ);
      int dofsPerElem = (1 << dim);
      //PERFORMANCE IMPROVEMENT: Better PreAllocation.
      if(activeNpes[lev + 1] > 1) {
        MatMPIAIJSetPreallocation(Pmat[lev], (dofsPerElem*dofsPerNode), PETSC_NULL, (dofsPerElem*dofsPerNode), PETSC_NULL);
      } else {
        MatSeqAIJSetPreallocation(Pmat[lev], (dofsPerElem*dofsPerNode), PETSC_NULL);
      }
    }
  }//end lev

  Vec rhs;
  Vec sol;

  assert(da[da.size() - 1] != NULL);
  DAGetGlobalVector(da[da.size() - 1], &rhs);
  DAGetGlobalVector(da[da.size() - 1], &sol);

  const unsigned int seed = (0x3456782  + (54763*globalRank));
  PetscRandom rndCtx;
  PetscRandomCreate(MPI_COMM_WORLD, &rndCtx);
  PetscRandomSetType(rndCtx, PETSCRAND48);
  PetscRandomSetSeed(rndCtx, seed);
  PetscRandomSeed(rndCtx);
  VecSetRandom(sol, rndCtx);
  PetscRandomDestroy(rndCtx);
  zeroBoundaries(da[da.size() - 1], sol);
  assert(Kmat[Kmat.size() - 1] != NULL);
  MatMult(Kmat[Kmat.size() - 1], sol, rhs);

  VecZeroEntries(sol);

  KSP ksp;
  PC pc;
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);
  KSPSetPreconditionerSide(ksp, PC_LEFT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCMG);
  PCMGSetLevels(pc, (da.size()), &(activeComms[0]));
  PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
  for(int lev = 1; lev < (da.size()); ++lev) {
    PCMGSetInterpolation(pc, lev, Pmat[lev - 1]);
  }//end lev
  KSPSetOperators(ksp, Kmat[Kmat.size() - 1], Kmat[Kmat.size() - 1], SAME_NONZERO_PATTERN);
  for(int lev = 0; lev < (Kmat.size()); ++lev) {
    KSP lksp;
    PC lpc;
    PCMGGetSmoother(pc, lev, &lksp);
    KSPSetType(lksp, KSPRICHARDSON);
    KSPSetPreconditionerSide(lksp, PC_LEFT);
    KSPRichardsonSetScale(lksp, 1.0);
    KSPGetPC(lksp, &lpc);
    PCSetType(lpc, PCSOR);
    PCSORSetOmega(lpc, 1.0);
    PCSORSetSymmetric(lpc, SOR_LOCAL_SYMMETRIC_SWEEP);
    PCSORSetIterations(lpc, 1, 2);
    KSPSetOperators(lksp, Kmat[lev], Kmat[lev], SAME_NONZERO_PATTERN);
  }//end lev
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetFromOptions(ksp);

  KSPSolve(ksp, rhs, sol);

  KSPDestroy(ksp);

  DARestoreGlobalVector(da[da.size() - 1], &rhs);
  DARestoreGlobalVector(da[da.size() - 1], &sol);

  for(int i = 0; i < da.size(); ++i) {
    if(da[i] != NULL) {
      DADestroy(da[i]);
    }
  }//end i
  da.clear();

  for(int i = 0; i < Kmat.size(); ++i) {
    if(Kmat[i] != NULL) {
      MatDestroy(Kmat[i]);
    }
  }//end i
  Kmat.clear();

  for(int i = 0; i < Pmat.size(); ++i) {
    if(Pmat[i] != NULL) {
      MatDestroy(Pmat[i]);
    }
  }//end i
  Pmat.clear();

  for(int i = 0; i < activeComms.size(); ++i) {
    if(activeComms[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&(activeComms[i]));
    }
  }//end i
  activeComms.clear();

  PetscFinalize();

  return 0;
}


