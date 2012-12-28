
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

  int npes;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

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

  //Compute Partition
  PetscInt Nx = 5;
  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &Nx, PETSC_NULL);

  //1-D case
  int px = npes;

  std::vector<PetscInt> partX;
  std::vector<PetscInt> scanX;
  std::vector<PetscInt> offsets;

  assert(px <= Nx);
  PetscInt avgX = Nx/px;
  PetscInt extraX = Nx%px; 
  partX.resize(px, avgX);
  for(int cnt = 0; cnt < extraX; ++cnt) {
    ++(partX[cnt]);
  }//end cnt

  offsets.resize(npes);
  for(int i = 0, p = 0; i < px; ++i, ++p) {
    offsets[p] = (partX[i]);
  }//end i

  for(int p = 1; p < npes; ++p) {
    offsets[p] += offsets[p - 1];
  }//end p

  for(int p = (npes - 1); p > 0; --p) {
    offsets[p] = offsets[p - 1];
  }//end p
  offsets[0] = 0;

  scanX.resize(px);
  scanX[0] = partX[0] - 1;
  for(int i = 1; i < px; ++i) {
    scanX[i] = scanX[i - 1] + partX[i];
  }//end i

  //Create DA
  DM da;
  if(dim == 1) {
    DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, Nx, dofsPerNode, 1, &(partX[0]), &da);
  } else if(dim == 2) {
    assert(false);
  } else {
    assert(false);
  }

  //Build Kmat
  Mat Kmat;

  //ComputeRHS
  Vec rhs;
  DMCreateGlobalVector(da, &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  //Build KSP
  PC pc;
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);
  KSPSetPCSide(ksp, PC_LEFT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCCHOLESKY);
  PCFactorSetShiftAmount(pc, 1.0e-13);
  PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
  KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
  KSPSetOperators(ksp, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
  KSPSetFromOptions(ksp);

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

  DMDestroy(&da);

  PetscFinalize();

  MPI_Finalize();

  return 0;
}






