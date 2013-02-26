
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscksp.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/mesh.h"
#include "gmg/include/assembly.h"
#include "gmg/include/boundary.h"
#include "gmg/include/intergrid.h"
#include "gmg/include/lsFitType3PC.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsNewTestType3", PETSC_NULL);

  PetscInt dim = 1; 
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

  std::vector<long long int> coeffsCK;
  read1DshapeFnCoeffs(K, coeffsCK);

  std::vector<long long int> coeffsC0;
  read1DshapeFnCoeffs(0, coeffsC0);

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
  computePartition(dim, Nz, Ny, Nx, partZ, partY, partX, offsets, 
      scanZ, scanY, scanX, activeNpes, print);
  assert(activeNpes[0] == npes);

  std::vector<MPI_Comm> activeComms;
  createActiveComms(activeNpes, activeComms);

  std::vector<DM> daCK;
  createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, daCK);

  std::vector<DM> daC0;
  createDA(dim, 1, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, daC0);

  std::vector<Mat> Kmat;
  buildKmat(Kmat, daCK, print);

  std::vector<Mat> reducedMat;
  buildKmat(reducedMat, daC0, false);

  assembleKmat(dim, Nz, Ny, Nx, Kmat, daCK, K, coeffsCK, factorialsList, print);

  assembleKmat(dim, Nz, Ny, Nx, reducedMat, daC0, 0, coeffsC0, factorialsList, false);

  correctKmat(Kmat, daCK, K);

  correctKmat(reducedMat, daC0, 0);

  Vec rhs;
  DMCreateGlobalVector(daCK[0], &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  VecSetRandom(sol, PETSC_NULL);
  double* solArr;
  VecGetArray(sol, &solArr);
  solArr[0] = 0;
  solArr[(Nx[0] - 1)*dofsPerNode] = 0;
  VecRestoreArray(sol, &solArr);

  MatMult(Kmat[0], sol, rhs);

  VecZeroEntries(sol);

  KSP ksp;
  PC pc;
  KSPCreate(activeComms[0], &ksp);
  KSPSetType(ksp, KSPFGMRES);
  //KSPSetPCSide(ksp, PC_RIGHT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCSHELL);
  setupLSfitType3PC(pc, Kmat[0], reducedMat[0], K, Nx[0], coeffsCK, coeffsC0);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetOperators(ksp, Kmat[0], Kmat[0], SAME_PRECONDITIONER);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 10);
  KSPSetOptionsPrefix(ksp, "outer_");
  KSPSetFromOptions(ksp);

  std::cout<<"Solving..."<<std::endl;

  KSPSolve(ksp, rhs, sol);

  VecDestroy(&rhs);
  VecDestroy(&sol);
  KSPDestroy(&ksp);
  destroyMat(Kmat);
  destroyMat(reducedMat);
  destroyDA(daCK); 
  destroyDA(daC0); 

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}




