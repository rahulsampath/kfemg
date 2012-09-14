
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscda.h"

int main(int argc, char *argv[]) {
  PetscInitialize(&argc, &argv, "optionsTestDA", PETSC_NULL);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);

  PetscInt dofsPerNode = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dofsPerNode", &dofsPerNode, PETSC_NULL);
  assert(dofsPerNode > 0);

  PetscInt Nx = 2; 
  PetscOptionsGetInt(PETSC_NULL, "-Nx", &Nx, PETSC_NULL);
  assert(Nx >= 1);

  PetscInt Ny = 1; 
  if(dim > 1) {
    PetscOptionsGetInt(PETSC_NULL, "-Ny", &Ny, PETSC_NULL);
  }
  assert(Ny >= 1);

  PetscInt Nz = 1; 
  if(dim > 2) {
    PetscOptionsGetInt(PETSC_NULL, "-Nz", &Nz, PETSC_NULL);
  }
  assert(Nz >= 1);

  int rank;
  int npes;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(!rank) {
    std::cout<<"Dim = "<<dim<<std::endl;
    std::cout<<"DofsPerNode = "<<dofsPerNode<<std::endl;
    std::cout<<"Nx = "<<Nx<<std::endl;
    std::cout<<"Ny = "<<Ny<<std::endl;
    std::cout<<"Nz = "<<Nz<<std::endl;
  }

  DA da;
  DACreate(MPI_COMM_WORLD, dim, DA_NONPERIODIC, DA_STENCIL_BOX, Nx, Ny, Nz,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dofsPerNode, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, (&da));

  PetscInt px;
  PetscInt py;
  PetscInt pz;
  DAGetInfo(da, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &px, &py, &pz, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL);

  assert(px >= 1);
  assert(py >= 1);
  assert(pz >= 1);
  assert(px <= Nx);
  assert(py <= Ny);
  assert(pz <= Nz);
  assert((px*py*pz) == npes);

  if(!rank) {
    std::cout<<"px = "<<px<<" py = "<<py<<" pz = "<<pz<<std::endl;
  }

  if(dim < 2) {
    assert(py == 1);
  }
  if(dim < 3) {
    assert(pz == 1);
  }

  int pk = rank/(px*py);
  int pj = (rank/px)%py;
  int pi = rank%px;

  assert(pi >= 0);
  assert(pj >= 0);
  assert(pk >= 0);
  assert(pi < px);
  assert(pj < py);
  assert(pk < pz);
  assert(((((pk*py)+ pj)*px) + pi) == rank);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  if(dim < 2) {
    assert(ys == 0);
    assert(ny == 1);
  }
  if(dim < 3) {
    assert(zs == 0);
    assert(nz == 1);
  }

  assert(nx > 0);
  assert(ny > 0);
  assert(nz > 0);

  DADestroy(da);

  PetscFinalize();

  return 0;

}


