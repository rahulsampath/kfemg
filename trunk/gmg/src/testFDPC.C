
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

  PetscInitialize(&argc, &argv, "optionsTestFDPC", PETSC_NULL);

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
  computePartition(dim, Nz, Ny, Nx, partZ, partY, partX, offsets, 
      scanZ, scanY, scanX, activeNpes, print);
  assert(activeNpes[activeNpes.size() - 1] == npes);

  std::vector<MPI_Comm> activeComms;
  createActiveComms(activeNpes, activeComms);

  std::vector<DM> da;
  createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, da);

  //Build Kmat
  std::vector<std::vector<std::vector<Mat> > > blkKmats;
  buildBlkKmats(blkKmats, da, activeComms, activeNpes);

  std::vector<Mat> Kmat;
  buildKmat(Kmat, da, print);

  //Matrix Assembly
  assembleBlkKmats(blkKmats, dim, dofsPerNode, Nz, Ny, Nx, partY, partX,
      offsets, da, K, coeffs, factorialsList);

  assembleKmat(dim, Nz, Ny, Nx, Kmat, da, K, coeffs, factorialsList, print);

  correctKmat(Kmat, da, K);

  correctBlkKmats(dim, blkKmats, da, partZ, partY, partX, offsets, K);

  std::vector<std::vector<Mat> > KhatMats;
  if(dim == 1) {
    createAll1DmatShells(K, activeComms, blkKmats, partX, KhatMats);
  } else {
    assert(false);
  }

  std::vector<std::vector<PC> > hatPc;
  createAll1DhatPc(partX, blkKmats, KhatMats, hatPc);

  int nlevels = da.size();
  Vec sol;
  Vec update;
  Vec rhs;
  Vec res;
  MatGetVecs(Kmat[nlevels - 1], &sol, &rhs);
  VecDuplicate(rhs, &res);
  VecDuplicate(sol, &update);

  VecSetRandom(sol, PETSC_NULL);
  MatMult(Kmat[nlevels - 1], sol, rhs);

  VecZeroEntries(sol);

  computeResidual(Kmat[nlevels - 1], sol, rhs, res);

  PetscReal initNorm;
  VecNorm(res, NORM_2, &initNorm);

  PCSetOperators(hatPc[nlevels - 2][K - 1], Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_PRECONDITIONER);
  PCApply(hatPc[nlevels - 2][K - 1], res, update);

  VecAXPY(sol, 1.0, update);

  computeResidual(Kmat[nlevels - 1], sol, rhs, res);

  PetscReal finalNorm;
  VecNorm(res, NORM_2, &finalNorm);

  std::cout<<"Init = "<<std::setprecision(13)<<initNorm<<std::endl;
  std::cout<<"Final = "<<std::setprecision(13)<<finalNorm<<std::endl;

  for(size_t i = 0; i < hatPc.size(); ++i) {
    for(size_t j = 0; j < (hatPc[i].size()); ++j) {
      PCFD1Ddata* data;
      PCShellGetContext(hatPc[i][j], (void**)(&data));
      destroyPCFD1Ddata(data);
      PCDestroy(&(hatPc[i][j]));
    }//end j
  }//end i
  hatPc.clear();

  VecDestroy(&update);
  VecDestroy(&sol);
  VecDestroy(&rhs);
  VecDestroy(&res);

  destroyMat(Kmat);
  destroyDA(da); 

  for(size_t i = 0; i < blkKmats.size(); ++i) {
    for(size_t j = 0; j < blkKmats[i].size(); ++j) {
      destroyMat(blkKmats[i][j]);
    }//end j
  }//end i

  for(size_t i = 0; i < KhatMats.size(); ++i) {
    for(size_t j = 0; j < KhatMats[i].size(); ++j) {
      Khat1Ddata* hatData;
      MatShellGetContext(KhatMats[i][j], &hatData);
      destroyKhat1Ddata(hatData);
      MatDestroy(&(KhatMats[i][j]));
    }//end j
  }//end i

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}




