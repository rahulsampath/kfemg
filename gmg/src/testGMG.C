
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
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
  PetscInt Nx = 17;
  PetscOptionsGetInt(PETSC_NULL, "-Nx", &Nx, PETSC_NULL);
  assert(Nx > 1);
  std::cout<<"Nx (Finest) = "<<Nx<<std::endl;
  PetscInt Ny = 1;
  if(dim > 1) {
    PetscOptionsGetInt(PETSC_NULL, "-Ny", &Ny, PETSC_NULL);
    assert(Ny > 1);
  }
  std::cout<<"Ny (Finest) = "<<Ny<<std::endl;
  PetscInt Nz = 1;
  if(dim > 2) {
    PetscOptionsGetInt(PETSC_NULL, "-Nz", &Nz, PETSC_NULL);
    assert(Nz > 1);
  }
  std::cout<<"Nz (Finest) = "<<Nz<<std::endl;
  PetscInt nlevels = 20;
  PetscOptionsGetInt(PETSC_NULL, "-nlevels", &nlevels, PETSC_NULL);
  std::cout<<"nlevels (Max) = "<<nlevels<<std::endl;

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);



  PetscFinalize();

  return 0;
}


