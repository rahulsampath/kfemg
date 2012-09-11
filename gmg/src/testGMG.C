
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscmg.h"
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
  createDA(da, activeComms, dofsPerNode, dim, Nz, Ny, Nx, MPI_COMM_WORLD);

  std::vector<Mat> Kmat(da.size(), NULL);
  for(int i = 0; i < da.size(); ++i) {
    if(da[i] != NULL) {
      DAGetMatrix(da[i], MATAIJ, &(Kmat[i]));
    }
  }//end i

  std::vector<Mat> Pmat((da.size() - 1), NULL);

  KSP ksp;
  PC pc;
  KSPCreate(MPI_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCMG);
  PCMGSetLevels(pc, (da.size()), &(activeComms[0]));
  PCMGSetType(pc, PC_MG_MULTIPLICATIVE);
  for(int lev = 1; lev < (da.size()); ++lev) {
    PCMGSetInterpolation(pc, lev, Pmat[lev - 1]);
  }//end lev
  KSPSetFromOptions(ksp);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

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


