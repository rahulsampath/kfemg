
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

  PetscInitialize(&argc, &argv, "optionsTestBlkKmat", PETSC_NULL);

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

  int nlevels = da.size();

  Mat fullMat = Kmat[nlevels - 1];
  Vec fullIn;
  Vec fullOut;
  MatGetVecs(fullMat, &fullIn, &fullOut);

  Vec blkIn;
  Vec blkOut;
  MatGetVecs(blkKmats[nlevels - 2][0][0], &blkIn, &blkOut);

  PetscInt blkSz;
  VecGetLocalSize(blkIn, &blkSz);

  VecSetRandom(blkIn, PETSC_NULL);

  for(int r = 0; r < dofsPerNode; ++r) {
    for(int c = 0; c < dofsPerNode; ++c) {
      if(c >= r) {
        MatMult(blkKmats[nlevels - 2][r][c - r], blkIn, blkOut);
      } else {
        MatMultTranspose(blkKmats[nlevels - 2][c][r - c], blkIn, blkOut);
      }

      VecZeroEntries(fullIn);

      double* blkArr;
      double* fullArr;
      VecGetArray(blkIn, &blkArr);
      VecGetArray(fullIn, &fullArr);
      for(int i = 0; i < blkSz; ++i) {
        fullArr[(dofsPerNode * i) + c] = blkArr[i];
      }//end i
      VecRestoreArray(blkIn, &blkArr);
      VecRestoreArray(fullIn, &fullArr);

      MatMult(fullMat, fullIn, fullOut);

      VecGetArray(blkOut, &blkArr);
      VecGetArray(fullOut, &fullArr);
      for(int i = 0; i < blkSz; ++i) {
        double diff = fullArr[(dofsPerNode * i) + r] - blkArr[i];
        assert(fabs(diff) < 1.0e-12);
      }//end i
      VecRestoreArray(blkOut, &blkArr);
      VecRestoreArray(fullOut, &fullArr);
    }//end c
  }//end r  

  correctKmat(Kmat, da, K);

  correctBlkKmats(dim, blkKmats, da, partZ, partY, partX, offsets, K);

  for(int r = 0; r < dofsPerNode; ++r) {
    for(int c = 0; c < dofsPerNode; ++c) {
      if(c >= r) {
        MatMult(blkKmats[nlevels - 2][r][c - r], blkIn, blkOut);
      } else {
        MatMultTranspose(blkKmats[nlevels - 2][c][r - c], blkIn, blkOut);
      }

      VecZeroEntries(fullIn);

      double* blkArr;
      double* fullArr;
      VecGetArray(blkIn, &blkArr);
      VecGetArray(fullIn, &fullArr);
      for(int i = 0; i < blkSz; ++i) {
        fullArr[(dofsPerNode * i) + c] = blkArr[i];
      }//end i
      VecRestoreArray(blkIn, &blkArr);
      VecRestoreArray(fullIn, &fullArr);

      MatMult(fullMat, fullIn, fullOut);

      VecGetArray(blkOut, &blkArr);
      VecGetArray(fullOut, &fullArr);
      for(int i = 0; i < blkSz; ++i) {
        double diff = fullArr[(dofsPerNode * i) + r] - blkArr[i];
        assert(fabs(diff) < 1.0e-12);
      }//end i
      VecRestoreArray(blkOut, &blkArr);
      VecRestoreArray(fullOut, &fullArr);
    }//end c
  }//end r  


  VecDestroy(&blkIn);
  VecDestroy(&blkOut);

  VecDestroy(&fullIn);
  VecDestroy(&fullOut);

  destroyMat(Kmat);
  destroyDA(da); 

  for(size_t i = 0; i < blkKmats.size(); ++i) {
    for(size_t j = 0; j < blkKmats[i].size(); ++j) {
      destroyMat(blkKmats[i][j]);
    }//end j
  }//end i

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}











