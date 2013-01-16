
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
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsTestMMS", PETSC_NULL);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);
  PetscInt K;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);

  int npes;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool print = (rank == 0);

  int dofsPerNode = getDofsPerNode(dim, K);

  if(print) {
    std::cout<<"Dim = "<<dim<<std::endl;
    std::cout<<"K = "<<K<<std::endl;
    std::cout<<"DofsPerNode = "<<dofsPerNode<<std::endl;
  }

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList); 

  std::vector<PetscInt> Nx;
  std::vector<PetscInt> Ny;
  std::vector<PetscInt> Nz;
  createGrids(dim, Nz, Ny, Nx, print);

  std::vector<std::vector<PetscInt> > partX;
  std::vector<std::vector<PetscInt> > partY;
  std::vector<std::vector<PetscInt> > partZ;
  std::vector<std::vector<PetscInt> > scanX;
  std::vector<std::vector<PetscInt> > scanY;
  std::vector<std::vector<PetscInt> > scanZ;
  std::vector<std::vector<PetscInt> > offsets;
  std::vector<int> activeNpes;
  computePartition(dim, Nz, Ny, Nx, partZ, partY, partX, offsets, scanZ, scanY, scanX, activeNpes);
  assert(activeNpes[activeNpes.size() - 1] == npes);

  std::vector<MPI_Comm> activeComms;
  createActiveComms(activeNpes, activeComms);

  std::vector<DM> da;
  createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, da);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(dim, dofsPerNode, Pmat, tmpCvec, da, activeComms, activeNpes); 

  computePmat(dim, factorialsList, Pmat, Nz, Ny, Nx, partZ, partY, partX, offsets,
      scanZ, scanY, scanX, dofsPerNode, coeffs, K);

  //Build Kmat
  std::vector<Mat> Kmat;
  buildKmat(Kmat, da, print);

  //Matrix Assembly
  assembleKmat(dim, Nz, Ny, Nx, Kmat, da, K, coeffs, factorialsList, print);

  Vec rhs;
  DMCreateGlobalVector(da[da.size() - 1], &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  //ComputeRHS
  computeRHS(da[da.size() - 1], coeffs, K, rhs);

  //Boundary corrections to rhs and sol. 
  VecZeroEntries(sol);
  setBoundaries(da[da.size() - 1], sol, K);
  VecScale(sol, -1.0);
  MatMultAdd(Kmat[Kmat.size() - 1], sol, rhs, rhs);
  setBoundaries(da[da.size() - 1], rhs, K);
  VecScale(sol, -1.0);

  correctKmat(Kmat, da, K);

  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;
  buildMGworkVecs(Kmat, mgSol, mgRhs, mgRes);

  //Build KSP
  PC pc;
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);
  KSPSetPCSide(ksp, PC_LEFT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCCHOLESKY);
  PCFactorSetShiftAmount(pc, 1.0e-12);
  PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetOperators(ksp, Kmat[Kmat.size() - 1], Kmat[Kmat.size() - 1], SAME_PRECONDITIONER);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
  KSPSetFromOptions(ksp);

  if(print) {
    std::cout<<"Solving..."<<std::endl;
  }

  KSPSolve(ksp, rhs, sol);

  //Compute Error
  long double err = computeError(da[da.size() - 1], sol, coeffs, K);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  VecDestroy(&rhs);
  VecDestroy(&sol);

  KSPDestroy(&ksp);

  destroyMat(Pmat);
  destroyVec(tmpCvec);
  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);
  destroyMat(Kmat);
  destroyDA(da); 

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}









