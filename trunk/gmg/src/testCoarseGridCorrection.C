
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
PetscLogEvent buildPmatEvent;
PetscLogEvent PmemEvent;
PetscLogEvent fillPmatEvent;
PetscLogEvent buildKmatEvent;
PetscLogEvent KmemEvent;
PetscLogEvent fillKmatEvent;
PetscLogEvent elemKmatEvent;
PetscLogEvent dirichletMatCorrectionEvent;
PetscLogEvent vCycleEvent;

void printVector(const unsigned int K, const unsigned int Nx, Vec in) {
  PetscScalar* inArr;
  VecGetArray(in, &inArr);
  for(int i = 0, cnt = 0; i < Nx; ++i) {
    for(int dx = 0; dx <= K; ++dx, ++cnt) {
      std::cout<<"("<<i<<","<<dx<<") = "<<(std::setprecision(13))<<(inArr[cnt])<<std::endl;
    }//end dx
  }//end i
  VecRestoreArray(in, &inArr);
}

void setInputVector(const unsigned int waveNum, const unsigned int waveDof,
    const unsigned int K, const unsigned int Nx, Vec in) {
  PetscScalar* inArr;
  VecGetArray(in, &inArr);

  for(int i = 0; i < ((K + 1)*Nx); ++i) {
    inArr[i] = 0.0;
  }//end i

  if((waveDof == 0) && (waveNum == 0)) {
    inArr[0] = 1.0;
  } else if((waveDof == 0) && (waveNum == (Nx - 1))) {
    inArr[((K + 1)*(Nx - 1))] = 1.0;
  } else {
    for(int i = 0; i < Nx; ++i) {
      double fac = (static_cast<double>(i*waveNum))/(static_cast<double>(Nx - 1));
      if(waveDof == 0) {
        inArr[((K + 1)*i)] = sin(fac*__PI__);
      } else {
        inArr[((K + 1)*i) + waveDof] = cos(fac*__PI__);
      }
    }//end i
    suppressSmallValues(((K + 1)*Nx), inArr);
  }

  VecRestoreArray(in, &inArr);
}

int main(int argc, char *argv[]) {
  PetscInitialize(&argc, &argv, "optionsTestCGC", PETSC_NULL);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);
  PetscInt finestNx = 9;
  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &finestNx, PETSC_NULL);
  PetscInt K = 0;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);
  PetscInt wNum = 0;
  PetscOptionsGetInt(PETSC_NULL, "-wNum", &wNum, PETSC_NULL);
  assert(wNum < finestNx);
  PetscInt wDof = 0;
  PetscOptionsGetInt(PETSC_NULL, "-wDof", &wDof, PETSC_NULL);
  assert(wDof <= K);

  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

  bool print = (globalRank == 0);

  int dofsPerNode = getDofsPerNode(dim, K);

  if(print) {
    std::cout<<"Dim = "<<dim<<std::endl;
    std::cout<<"Fine-Nx = "<<finestNx<<std::endl;
    std::cout<<"K = "<<K<<std::endl;
    std::cout<<"wNum = "<<wNum<<std::endl;
    std::cout<<"wDof = "<<wDof<<std::endl;
    std::cout<<"DofsPerNode = "<<dofsPerNode<<std::endl;
  }

  std::vector<DA> da;
  std::vector<PetscInt> Nx;
  std::vector<PetscInt> Ny;
  std::vector<PetscInt> Nz;
  std::vector<MPI_Comm> activeComms;
  std::vector<int> activeNpes;
  std::vector<std::vector<PetscInt> > partZ;
  std::vector<std::vector<PetscInt> > partY;
  std::vector<std::vector<PetscInt> > partX;
  std::vector<std::vector<int> > offsets;
  std::vector<std::vector<int> > scanLz;
  std::vector<std::vector<int> > scanLy;
  std::vector<std::vector<int> > scanLx;
  createDA(da, activeComms, activeNpes, dofsPerNode, dim, Nz, Ny, Nx, partZ, partY, partX,
      offsets, scanLz, scanLy, scanLx, MPI_COMM_WORLD, print);

  assert(da[da.size() - 1] != NULL);

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList); 

  std::vector<Mat> Kmat;
  buildKmat(factorialsList, Kmat, da, activeComms, activeNpes, dim, dofsPerNode, coeffs, K,
      partZ, partY, partX, offsets, print);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(factorialsList, Pmat, tmpCvec, da, activeComms, activeNpes, dim, dofsPerNode, coeffs, K, Nz, Ny, Nx,
      partZ, partY, partX, offsets, scanLz, scanLy, scanLx, print);

  Vec rhs;
  DACreateGlobalVector(da[da.size() - 1], &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol);

  Vec res;
  VecDuplicate(rhs, &res);

  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;

  buildMGworkVecs(Kmat, mgSol, mgRhs, mgRes);

  mgSol[Kmat.size() - 1] = sol;
  mgRhs[Kmat.size() - 1] = rhs;
  mgRes[Kmat.size() - 1] = res;

  KSP ksp;
  PC pc;
  KSPCreate(activeComms[0], &ksp);
  KSPGetPC(ksp, &pc);
  KSPSetType(ksp, KSPPREONLY);
  KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
  PCSetType(pc, PCLU);
  KSPSetOperators(ksp, Kmat[0], Kmat[0], SAME_NONZERO_PATTERN);

  VecZeroEntries(rhs);

  setInputVector(wNum, wDof, K, finestNx, sol);
  zeroBoundaries(da[1], sol);

  PetscReal norm;
  VecNorm(sol, NORM_INFINITY, &norm);
  if(print) {
    std::cout<<"Initial Max Norm = "<<std::setprecision(15)<<norm<<std::endl;
  }

  computeResidual(Kmat[1], mgSol[1], mgRhs[1], mgRes[1]);
  applyRestriction(Pmat[0], tmpCvec[0], mgRes[1], mgRhs[0]);
  //VecZeroEntries(mgSol[0]);
  KSPSolve(ksp, mgRhs[0], mgSol[0]);
  applyProlongation(Pmat[0], tmpCvec[0], mgSol[0], mgRes[1]);
  VecAXPY(mgSol[1], 1.0, mgRes[1]);

  VecNorm(sol, NORM_INFINITY, &norm);
  if(print) {
    std::cout<<"Final Max Norm = "<<std::setprecision(15)<<norm<<std::endl;
  }

  printVector(K, finestNx, sol);

  mgSol[Kmat.size() - 1] = NULL;
  mgRhs[Kmat.size() - 1] = NULL;
  mgRes[Kmat.size() - 1] = NULL;

  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);

  VecDestroy(rhs);
  VecDestroy(res);
  VecDestroy(sol);

  KSPDestroy(ksp);

  destroyDA(da);

  destroyVec(tmpCvec);

  destroyMat(Pmat);

  destroyMat(Kmat);

  destroyComms(activeComms);

  PetscFinalize();

  return 0;
}



