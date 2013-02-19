
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
#include "gmg/include/hatPC.h"

void applyFDapprox(Vec in, Vec out) {
  PetscInt sz;
  VecGetSize(in, &sz);
  //double* inArr;
  //double* outArr;
  //  VecGetArray(in, &inArr);
  //  VecGetArray(out, &outArr);

  VecZeroEntries(out);

  /*
  //First Order
  for(int i = 0; i < (sz - 1); ++i) {
  outArr[i] = (inArr[i + 1] - inArr[i])/2.0;
  }//end i
  outArr[sz - 1] = (inArr[sz - 1] - inArr[sz - 2])/2.0;
  */

  /*
  //Second Order
  outArr[0] = -((3.0 * inArr[0]) - (4.0 * inArr[1]) + inArr[2])/4.0;
  for(int i = 1; i < (sz - 1); ++i) {
  outArr[i] = (inArr[i + 1] - inArr[i - 1])/4.0;
  }//end i
  outArr[sz - 1] = ((3.0 * inArr[sz - 1]) - (4.0 * inArr[sz - 2]) + inArr[sz - 3])/4.0;
  */

  /*
     int nx = sz;
  //Fourth Order
  outArr[0] = -((25.0 * inArr[0]) - (48.0 * inArr[1]) + (36.0 * inArr[2]) - (16.0 * inArr[3]) + (3.0 * inArr[4]))/24.0;
  outArr[1] = -((25.0 * inArr[1]) - (48.0 * inArr[2]) + (36.0 * inArr[3]) - (16.0 * inArr[4]) + (3.0 * inArr[5]))/24.0;
  for(int i = 2; i < (nx - 2); ++i) {
  outArr[i] = (-inArr[i + 2] + (8.0 * inArr[i + 1]) - (8.0 * inArr[i - 1]) + inArr[i - 2])/24.0;
  }//end i
  outArr[nx - 2] = ((25.0 * inArr[nx - 2]) - (48.0 * inArr[nx - 3]) + (36.0 * inArr[nx - 4]) - (16.0 * inArr[nx - 5]) + (3.0 * inArr[nx - 6]))/24.0;
  outArr[nx - 1] = ((25.0 * inArr[nx - 1]) - (48.0 * inArr[nx - 2]) + (36.0 * inArr[nx - 3]) - (16.0 * inArr[nx - 4]) + (3.0 * inArr[nx - 5]))/24.0;
  */

  //  VecRestoreArray(in, &inArr);
  //  VecRestoreArray(out, &outArr);
}

PetscErrorCode myMatMult(Mat mat, Vec in, Vec out) {
  Mat Kmat;
  MatShellGetContext(mat, &Kmat);
  Vec inPrime;
  VecDuplicate(in, &inPrime);
  PetscInt sz;
  VecGetSize(in, &sz);
  applyFDapprox(in, inPrime);
  Vec Kin, Kout;
  MatGetVecs(Kmat, &Kin, &Kout);
  double* fArr;
  double* b0Arr; 
  double* b1Arr;
  VecGetArray(Kin, &fArr);
  VecGetArray(in, &b0Arr);
  VecGetArray(inPrime, &b1Arr);
  for(int i = 0; i < sz; ++i) {
    fArr[(2*i)] = b0Arr[i];
    fArr[(2*i) + 1] = b1Arr[i];
  }//end i
  VecRestoreArray(Kin, &fArr);
  VecRestoreArray(in, &b0Arr);
  VecRestoreArray(inPrime, &b1Arr);
  MatMult(Kmat, Kin, Kout);
  VecGetArray(Kout, &fArr);
  VecGetArray(out, &b0Arr);
  for(int i = 0; i < sz; ++i) {
    b0Arr[i] = (fArr[(2*i)] + fArr[(2*i) + 1]);
  }//end i
  VecRestoreArray(Kout, &fArr);
  VecRestoreArray(out, &b0Arr);
  VecDestroy(&Kin);
  VecDestroy(&Kout);
  VecDestroy(&inPrime);
  return 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsMyTest", PETSC_NULL);

  PetscInt dim = 1; 
  PetscInt K = 1;

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

  //MyPC
  {
    Mat Khat;
    MatCreateShell(PETSC_COMM_WORLD, blkSz, blkSz, PETSC_DETERMINE, PETSC_DETERMINE, Kmat[nlevels - 1], &Khat);
    MatShellSetOperation(Khat, MATOP_MULT, (void(*)(void))(&myMatMult));

    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetType(ksp, KSPFGMRES);
    KSPSetPCSide(ksp, PC_RIGHT);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCNONE);
    KSPSetOperators(ksp, Khat, Khat, SAME_PRECONDITIONER);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 1);
    KSPSetOptionsPrefix(ksp, "inner_");
    KSPSetFromOptions(ksp);

    Vec blkSol;
    VecDuplicate(blkRes, &blkSol);

    VecZeroEntries(blkSol);
    double* arr1;
    double* arr2;
    double* arr3;
    VecGetArray(res, &arr1);
    VecGetArray(blkSol, &arr2);
    VecGetArray(blkRes, &arr3);
    arr2[0] = arr1[0];
    arr2[blkSz - 1] = arr1[2*(blkSz - 1)];
    for(int i = 0; i < blkSz; ++i) {
      arr3[i] = arr1[2*i] + arr1[(2*i) + 1];
    }//end i
    VecRestoreArray(blkRes, &arr3);
    VecRestoreArray(blkSol, &arr2);
    VecRestoreArray(res, &arr1);

    KSPSolve(ksp, blkRes, blkSol);
    applyFDapprox(blkSol, blkRes);
    VecGetArray(update, &arr1);
    VecGetArray(blkSol, &arr2);
    VecGetArray(blkRes, &arr3);
    for(int i = 0; i < blkSz; ++i) {
      arr1[(2*i)] = arr2[i];
      arr1[(2*i) + 1] = arr3[i];
    }//end i
    VecRestoreArray(blkRes, &arr3);
    VecRestoreArray(blkSol, &arr2);
    VecRestoreArray(update, &arr1);
    VecDestroy(&blkSol);

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
  }

  //Plain
  {
    KSP ksp;
    PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetType(ksp, KSPCG);
    KSPSetPCSide(ksp, PC_LEFT);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCNONE);
    KSPSetOperators(ksp, Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_PRECONDITIONER);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 1);
    KSPSetOptionsPrefix(ksp, "plain_");
    KSPSetFromOptions(ksp);

    VecZeroEntries(sol);
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
  }

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

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}




