
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

  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

  bool print = (globalRank == 0);

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

  std::vector<PetscInt> partX;
  std::vector<PetscInt> scanX;
  std::vector<PetscInt> offsets;

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
  KSP ksp;

  KSPSolve(ksp, rhs, sol);

  //Compute Error
  long double err;

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






