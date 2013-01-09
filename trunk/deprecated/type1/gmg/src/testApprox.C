
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscksp.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"

PetscLogEvent createDAevent;
PetscLogEvent errEvent;
PetscLogEvent rhsEvent;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsTestApprox", PETSC_NULL);

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

  std::vector<DM> da;
  std::vector<PetscInt> Nx;
  std::vector<PetscInt> Ny;
  std::vector<PetscInt> Nz;
  std::vector<MPI_Comm> activeComms;
  std::vector<int> activeNpes;
  std::vector<std::vector<PetscInt> > partZ;
  std::vector<std::vector<PetscInt> > partY;
  std::vector<std::vector<PetscInt> > partX;
  std::vector<std::vector<PetscInt> > offsets;
  std::vector<std::vector<PetscInt> > scanLz;
  std::vector<std::vector<PetscInt> > scanLy;
  std::vector<std::vector<PetscInt> > scanLx;
  createDA(da, activeComms, activeNpes, dofsPerNode, dim, Nz, Ny, Nx, partZ, partY, partX,
      offsets, scanLz, scanLy, scanLx, MPI_COMM_WORLD, print);

  assert(da[da.size() - 1] != NULL);

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  Vec sol;
  DMCreateGlobalVector(da[da.size() - 1], &sol);

  setSolution(da[da.size() - 1], sol, K);

  long double err = computeError(da[da.size() - 1], sol, coeffs, K);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  VecDestroy(&sol);

  destroyDA(da);

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}


