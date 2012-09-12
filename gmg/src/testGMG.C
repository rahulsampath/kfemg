
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
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

  int dofsPerNode = getDofsPerNode(dim, K);

  std::vector<PetscInt> Nx;
  std::vector<PetscInt> Ny;
  std::vector<PetscInt> Nz;
  createGridSizes(dim, Nz, Ny, Nx);

  std::vector<DA> da;
  std::vector<MPI_Comm> activeComms;
  std::vector<int> activeNpes;
  createDA(da, activeComms, activeNpes, dofsPerNode, dim, Nz, Ny, Nx, MPI_COMM_WORLD);

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<Mat> Kmat;
  buildKmat(Kmat, da);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(Pmat, tmpVec, da, activeComms, activeNpes, dim, dofsPerNode);

  assert(da[da.size() - 1] != NULL);

  Vec rhs;
  DACreateGlobalVector(da[da.size() - 1], &rhs);

  const unsigned int seed = (0x3456782  + (54763*globalRank));
  computeRandomRHS(da[da.size() - 1], Kmat[Kmat.size() - 1], rhs, seed);

  Vec sol;
  DACreateGlobalVector(da[da.size() - 1], &sol);
  VecZeroEntries(sol);

  KSP ksp;
  createSolver(ksp, Kmat, Pmat, activeComms);

  KSPSolve(ksp, rhs, sol);

  KSPDestroy(ksp);

  VecDestroy(rhs);
  VecDestroy(sol);

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

  for(int i = 0; i < tmpCvec.size(); ++i) {
    if(tmpCvec[i] != NULL) {
      VecDestroy(tmpCvec[i]);
    }
  }//end i
  tmpCvec.clear();

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


