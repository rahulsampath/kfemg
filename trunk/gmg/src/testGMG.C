
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
  PetscInitialize(&argc, &argv, "optionsTestGMG", PETSC_NULL);
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

  assert(da[da.size() - 1] != NULL);

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<Mat> Kmat;
  buildKmat(Kmat, da, coeffs, K);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(Pmat, tmpCvec, da, activeComms, activeNpes, dim, dofsPerNode, coeffs, K);

  std::vector<KSP> ksp;
  createKSP(ksp, Kmat, activeComms);

  Vec rhs;
  DACreateGlobalVector(da[da.size() - 1], &rhs);

  const unsigned int seed = (0x3456782  + (54763*globalRank));
  computeRandomRHS(da[da.size() - 1], Kmat[Kmat.size() - 1], rhs, seed);

  Vec sol;
  VecDuplicate(rhs, &sol);

  Vec res;
  VecDuplicate(rhs, &res);

  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;

  buildMGworkVecs(Kmat, mgSol, mgRhs, mgRes);

  mgSol[Kmat.size() - 1] = sol;
  mgRhs[Kmat.size() - 1] = rhs;
  mgRes[Kmat.size() - 1] = res;

  VecZeroEntries(sol);

  computeResidual(Kmat[Kmat.size() - 1], sol, rhs, res);
  PetscReal initialResNorm;
  VecNorm(res, NORM_2, &initialResNorm);
  std::cout<<"Initial Residual Norm = "<<initialResNorm<<std::endl;

  int iter = 0;
  while(iter < maxVcycles) {
    applyVcycle((Kmat.size() - 1), Kmat, Pmat, tmpCvec, ksp, mgSol, mgRhs, mgRes);
    ++iter;
    computeResidual(Kmat[Kmat.size() - 1], sol, rhs, res);
    PetscReal currResNorm;
    VecNorm(res, NORM_2, &currResNorm);
    std::cout<<"Iter = "<<iter<<" ResNorm = "<<currResNorm<<std::endl;
    if(currResNorm < aTol) {
      std::cout<<"Converged due to ATOL."<<std::endl;
      break;
    }
    if(currResNorm < (rTol*initialResNorm)) {
      std::cout<<"Converged due to RTOL."<<std::endl;
      break;
    }
  }
  std::cout<<"Number of V-cycles = "<<iter<<std::endl;

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


