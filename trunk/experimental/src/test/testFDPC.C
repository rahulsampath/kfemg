
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
#include "gmg/include/hatPC.h"
#include "gmg/include/assembly.h"
#include "gmg/include/boundary.h"

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

  int nlevels = da.size();
 //MatZeroEntries(blkKmats[nlevels - 2][0][1]);

  std::vector<std::vector<Mat> > KhatMats;
  if(dim == 1) {
    createAll1DmatShells(K, activeComms, blkKmats, partX, KhatMats);
  } else {
    assert(false);
  }

  std::vector<std::vector<PC> > hatPc;
  createAll1DhatPc(partX, blkKmats, KhatMats, hatPc);

  Vec sol;
  Vec update;
  Vec rhs;
  Vec res;
  MatGetVecs(Kmat[nlevels - 1], &sol, &rhs);
  VecDuplicate(rhs, &res);
  VecDuplicate(sol, &update);

  Vec blkRes;
  MatGetVecs(blkKmats[nlevels - 2][0][0], &blkRes, PETSC_NULL);
  PetscInt blkSz;
  VecGetLocalSize(blkRes, &blkSz);

  VecSetRandom(sol, PETSC_NULL);
  MatMult(Kmat[nlevels - 1], sol, rhs);

  VecZeroEntries(sol);
  double* rhsArr;
  double* solArr;
  VecGetArray(rhs, &rhsArr);
  VecGetArray(sol, &solArr);
  solArr[0] = rhsArr[0];
  solArr[(2*Nx[nlevels - 1]) - 2] = rhsArr[(2*Nx[nlevels - 1]) - 2]; 
  VecRestoreArray(rhs, &rhsArr);
  VecRestoreArray(sol, &solArr);

  std::cout<<"Using HatPC: "<<std::endl;
  computeResidual(Kmat[nlevels - 1], sol, rhs, res);
  PetscReal initNorm;
  VecNorm(res, NORM_2, &initNorm);
  std::cout<<"Full Init = "<<std::setprecision(13)<<initNorm<<std::endl;
  for(int d = 0; d < dofsPerNode; ++d) {
    double* blkArr;
    double* fullArr;
    VecGetArray(res, &fullArr);
    VecGetArray(blkRes, &blkArr);
    for(int i = 0; i < blkSz; ++i) {
      blkArr[i] = fullArr[(dofsPerNode * i) + d];
    }//end i
    VecRestoreArray(res, &fullArr);
    VecRestoreArray(blkRes, &blkArr);
    PetscReal norm;
    VecNorm(blkRes, NORM_2, &norm);
    std::cout<<"Init["<<d<<"]= "<<std::setprecision(13)<<norm<<std::endl;
    //VecView(blkRes, PETSC_VIEWER_STDOUT_WORLD);
  }//end d

  PCSetOperators(hatPc[nlevels - 2][K - 1], Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_PRECONDITIONER);
  PCApply(hatPc[nlevels - 2][K - 1], res, update);

  VecAXPY(sol, 1.0, update);

  computeResidual(Kmat[nlevels - 1], sol, rhs, res);
  PetscReal finalNorm;
  VecNorm(res, NORM_2, &finalNorm);
  std::cout<<"Full Final = "<<std::setprecision(13)<<finalNorm<<std::endl;
  for(int d = 0; d < dofsPerNode; ++d) {
    double* blkArr;
    double* fullArr;
    VecGetArray(res, &fullArr);
    VecGetArray(blkRes, &blkArr);
    for(int i = 0; i < blkSz; ++i) {
      blkArr[i] = fullArr[(dofsPerNode * i) + d];
    }//end i
    VecRestoreArray(res, &fullArr);
    VecRestoreArray(blkRes, &blkArr);
    PetscReal norm;
    VecNorm(blkRes, NORM_2, &norm);
    std::cout<<"Final["<<d<<"]= "<<std::setprecision(13)<<norm<<std::endl;
    //VecView(blkRes, PETSC_VIEWER_STDOUT_WORLD);
  }//end d

  KSP ksp;
  PC pc;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);
  KSPSetPCSide(ksp, PC_LEFT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCNONE);
  KSPSetOperators(ksp, Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_PRECONDITIONER);
  KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 1);
  KSPSetOptionsPrefix(ksp, "plain_");
  KSPSetFromOptions(ksp);

  KSPSolve(ksp, rhs, sol);

  computeResidual(Kmat[nlevels - 1], sol, rhs, res);
  for(int d = 0; d < dofsPerNode; ++d) {
    double* blkArr;
    double* fullArr;
    VecGetArray(res, &fullArr);
    VecGetArray(blkRes, &blkArr);
    for(int i = 0; i < blkSz; ++i) {
      blkArr[i] = fullArr[(dofsPerNode * i) + d];
    }//end i
    VecRestoreArray(res, &fullArr);
    VecRestoreArray(blkRes, &blkArr);
    PetscReal norm;
    VecNorm(blkRes, NORM_2, &norm);
    std::cout<<"Plain["<<d<<"]= "<<std::setprecision(13)<<norm<<std::endl;
    //VecView(blkRes, PETSC_VIEWER_STDOUT_WORLD);
  }//end d

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
  VecDestroy(&blkRes);

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




