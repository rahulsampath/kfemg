
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

  std::vector<MPI_Comm> activeComms;
  createActiveComms(activeNpes, activeComms);

  std::vector<DM> da;
  createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, da);

  //Build Kmat
  Mat Kmat;
  DMCreateMatrix(da, MATAIJ, &Kmat);

  PetscInt sz;
  MatGetSize(Kmat, &sz, PETSC_NULL);
  if(print) {
    std::cout<<"Kmat Size = "<<sz<<std::endl;
  }

  std::vector<std::vector<long double> > elemMat;
  if(dim == 1) {
    long double hx = 1.0L/(static_cast<long double>(Nx - 1));
    createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, print);
  } else if(dim == 2) {
    long double hx = 1.0L/(static_cast<long double>(Nx - 1));
    long double hy = 1.0L/(static_cast<long double>(Ny - 1));
    createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, print);
  } else {
    long double hx = 1.0L/(static_cast<long double>(Nx - 1));
    long double hy = 1.0L/(static_cast<long double>(Ny - 1));
    long double hz = 1.0L/(static_cast<long double>(Nz - 1));
    createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMat, print);
  }

  Vec rhs;
  DMCreateGlobalVector(da, &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  //ComputeRHS
  computeRHS(da, coeffs, K, rhs);

  //Matrix Assembly
  computeKmat(Kmat, da, elemMat);

  //ModifyRHS 
  VecZeroEntries(sol);
  setBoundaries(da, sol, K);
  VecScale(sol, -1.0);
  MatMultAdd(Kmat, sol, rhs, rhs);
  setBoundaries(da, rhs, K);

  dirichletMatrixCorrection(Kmat, da, K);

  VecScale(sol, -1.0);
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
  KSPSetOperators(ksp, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
  KSPSetFromOptions(ksp);

  if(print) {
    std::cout<<"Solving..."<<std::endl;
  }

  KSPSolve(ksp, rhs, sol);

  //Compute Error
  long double err = computeError(da, sol, coeffs, K);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  VecDestroy(&rhs);
  VecDestroy(&sol);

  KSPDestroy(&ksp);

  MatDestroy(&Kmat);

  destroyDA(da); 

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}









