
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
  PetscInt maxVcycles = 1;
  PetscOptionsGetInt(PETSC_NULL, "-maxVcycles", &maxVcycles, PETSC_NULL);
  PetscReal rTol = 1.0e-12;
  PetscOptionsGetReal(PETSC_NULL, "-rTol", &rTol, PETSC_NULL);
  PetscReal aTol = 1.0e-12;
  PetscOptionsGetReal(PETSC_NULL, "-aTol", &aTol, PETSC_NULL);

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
  buildPmat(Pmat, tmpCvec, da, activeComms, activeNpes, dim, dofsPerNode);

  assert(da[da.size() - 1] != NULL);

  Vec rhs;
  DACreateGlobalVector(da[da.size() - 1], &rhs);

  const unsigned int seed = (0x3456782  + (54763*globalRank));
  computeRandomRHS(da[da.size() - 1], Kmat[Kmat.size() - 1], rhs, seed);

  Vec sol;
  VecDuplicate(rhs, &sol);
  VecZeroEntries(sol);

  Vec res;
  VecDuplicate(rhs, &res);

  std::vector<KSP> ksp;
  createKSP(ksp, Kmat, activeComms);

  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;

  mgSol.resize(Kmat.size(), NULL);
  mgRhs.resize(Kmat.size(), NULL);
  mgRes.resize(Kmat.size(), NULL);
  for(int i = 0; i < (Kmat.size() - 1); ++i) {
    if(Kmat[i] != NULL) {
      MatGetVecs(Kmat[i], &(mgSol[i]), &(mgRhs[i]));
      VecDuplicate(mgRhs[i], &(mgRes[i]));
    }
  }//end i
  mgSol[Kmat.size() - 1] = sol;
  mgRhs[Kmat.size() - 1] = rhs;
  mgRes[Kmat.size() - 1] = res;

  computeResidual(Kmat[Kmat.size() - 1], sol, rhs, res);
  PetscReal initialResNorm;
  VecNorm(res, NORM_2, &initialResNorm);
  std::cout<<"Initial Residual Norm = "<<initialResNorm<<std::endl;


  mgSol[Kmat.size() - 1] = NULL;
  mgRhs[Kmat.size() - 1] = NULL;
  mgRes[Kmat.size() - 1] = NULL;

  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);

  destroyKSP(ksp);

  VecDestroy(rhs);
  VecDestroy(res);
  VecDestroy(sol);

  destroyDA(da);

  destroyVec(tmpCvec);

  destroyMat(Pmat);

  destroyMat(Kmat);

  destroyComms(activeComms);

  PetscFinalize();

  return 0;
}


