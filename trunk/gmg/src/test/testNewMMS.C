
#include "petsc.h"
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/mesh.h"
#include "gmg/include/mms.h"
#include "gmg/include/newRtgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/assembly.h"
#include "gmg/include/intergrid.h"
#include "common/include/commonUtils.h"
#include <iomanip>
#include <iostream>

#ifdef DEBUG
#include <cassert>
#endif

PetscClassId gmgCookie;
PetscLogEvent meshEvent;
PetscLogEvent buildPmatEvent;
PetscLogEvent buildKmatEvent;
PetscLogEvent rhsEvent;
PetscLogEvent solverSetupEvent;
PetscLogEvent solverApplyEvent;
PetscLogEvent errEvent;
PetscLogEvent cleanupEvent;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsMMS", PETSC_NULL);

  PetscClassIdRegister("GMG", &gmgCookie);
  PetscLogEventRegister("Mesh", gmgCookie, &meshEvent);
  PetscLogEventRegister("BuildPmat", gmgCookie, &buildPmatEvent);
  PetscLogEventRegister("BuildKmat", gmgCookie, &buildKmatEvent);
  PetscLogEventRegister("RHS", gmgCookie, &rhsEvent);
  PetscLogEventRegister("SolverSetup", gmgCookie, &solverSetupEvent);
  PetscLogEventRegister("SolverApply", gmgCookie, &solverApplyEvent);
  PetscLogEventRegister("Error", gmgCookie, &errEvent);
  PetscLogEventRegister("Cleanup", gmgCookie, &cleanupEvent);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
#ifdef DEBUG
  assert(dim > 0);
  assert(dim <= 3);
#endif
  PetscInt K = 1;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);

  int npes;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool print = (rank == 0);

  if(print) {
    std::cout<<"Dim = "<<dim<<std::endl;
    std::cout<<"K = "<<K<<std::endl;
  }

  PetscLogEventBegin(meshEvent, 0, 0, 0, 0);

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
  int nlevels = activeNpes.size();
#ifdef DEBUG
  assert(activeNpes[nlevels - 1] == npes);
#endif

  std::vector<MPI_Comm> activeComms;
  createActiveComms(activeNpes, activeComms);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList); 

  char prefix[200] = "";
  PetscOptionsGetString(PETSC_NULL, "-coeffsDirPath", prefix, 200, PETSC_NULL);

  std::vector<std::vector<long long int> > coeffs(K + 1);
  read1DshapeFnCoeffs(K, prefix, coeffs[K]);
  if(nlevels > 1) {
    for(int k = 0; k < K; ++k) {
      read1DshapeFnCoeffs(k, prefix, coeffs[k]);
    }//end k
  }

  std::vector<std::vector<DM> > da(K + 1);
  {
    int dofsPerNode = getDofsPerNode(dim, K);
    createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, 
        partX, activeNpes, activeComms, da[K]);
  }
  if(nlevels > 1) {
    for(int k = 0; k < K; ++k) {
      int dofsPerNode = getDofsPerNode(dim, k);
      createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, 
          partX, activeNpes, activeComms, da[k]);
    }//end k
  }

  PetscLogEventEnd(meshEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildPmatEvent, 0, 0, 0, 0);

  std::vector<std::vector<Mat> > Pmat(K + 1);
  std::vector<std::vector<Vec> > tmpCvec(K + 1);
  {
    int dofsPerNode = getDofsPerNode(dim, K);
    buildPmat(dim, dofsPerNode, Pmat[K], tmpCvec[K], da[K], activeComms, activeNpes); 
    computePmat(dim, factorialsList, Pmat[K], Nz, Ny, Nx, partZ, partY, partX, offsets,
        scanZ, scanY, scanX, dofsPerNode, coeffs[K], K);
  }
  for(int k = 0; k < K; ++k) {
    int dofsPerNode = getDofsPerNode(dim, k);
    buildPmat(dim, dofsPerNode, Pmat[k], tmpCvec[k], da[k], activeComms, activeNpes); 
    computePmat(dim, factorialsList, Pmat[k], Nz, Ny, Nx, partZ, partY, partX, offsets,
        scanZ, scanY, scanX, dofsPerNode, coeffs[k], k);
  }//end k

  PetscLogEventEnd(buildPmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  std::vector<std::vector<Mat> > Kmat(K + 1);
  buildKmat(Kmat[K], da[K], print);
  assembleKmat(dim, Nz, Ny, Nx, Kmat[K], da[K], K, coeffs[K], factorialsList, print);
  if(nlevels > 1) {
    for(int k = 0; k < K; ++k) {
      buildKmat(Kmat[k], da[k], print);
      assembleKmat(dim, Nz, Ny, Nx, Kmat[k], da[k], k, coeffs[k], factorialsList, print);
    }//end k
  }

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(rhsEvent, 0, 0, 0, 0);

  Vec sol;
  Vec rhs;
  MatGetVecs(Kmat[K][nlevels - 1], &sol, &rhs);

  computeRHS(da[K][nlevels - 1], coeffs[K], K, rhs);
  VecZeroEntries(sol);
  setBoundaries(da[K][nlevels - 1], sol, K);
  VecScale(sol, -1.0);
  MatMultAdd(Kmat[K][nlevels - 1], sol, rhs, rhs);
  VecScale(sol, -1.0);
  makeBoundariesConsistent(da[K][nlevels - 1], sol, rhs, K);

  PetscLogEventEnd(rhsEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  correctKmat(Kmat[K], da[K], K);
  if(nlevels > 1) {
    for(int k = 0; k < K; ++k) {
      correctKmat(Kmat[k], da[k], k);
    }//end k
  }

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(solverSetupEvent, 0, 0, 0, 0);

  KSP ksp;
  PC pc;
  KSPCreate(activeComms[nlevels - 1], &ksp);
  KSPGetPC(ksp, &pc);
  if(nlevels == 1) {
    KSPSetType(ksp, KSPCG);
    KSPSetPCSide(ksp, PC_LEFT);
    PCSetType(pc, PCCHOLESKY);
    PCFactorSetShiftAmount(pc, 1.0e-12);
    PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
  } else {
    KSPSetType(ksp, KSPFGMRES);
    KSPSetPCSide(ksp, PC_RIGHT);
    setupNewRTG(pc, K, (nlevels - 1), da, coeffs, Kmat, Pmat, tmpCvec);
  }
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetOperators(ksp, Kmat[K][nlevels - 1], Kmat[K][nlevels - 1], SAME_PRECONDITIONER);
  KSPSetTolerances(ksp, 1.0e-10, 1.0e-10, PETSC_DEFAULT, 500);
  KSPDefaultConvergedSetUIRNorm(ksp);
  KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
  KSPSetOptionsPrefix(ksp, "outer_");
  KSPSetFromOptions(ksp);

  PetscLogEventEnd(solverSetupEvent, 0, 0, 0, 0);

  if(print) {
    std::cout<<"Solving..."<<std::endl;
  }

  PetscLogEventBegin(solverApplyEvent, 0, 0, 0, 0);

  KSPSolve(ksp, rhs, sol);

  PetscLogEventEnd(solverApplyEvent, 0, 0, 0, 0);

  PetscLogEventBegin(errEvent, 0, 0, 0, 0);

  long double err = computeError(da[K][nlevels - 1], sol, coeffs[K], K);

  PetscLogEventEnd(errEvent, 0, 0, 0, 0);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  PetscLogEventBegin(cleanupEvent, 0, 0, 0, 0);

  VecDestroy(&rhs);
  VecDestroy(&sol);
  KSPDestroy(&ksp);
  for(int k = 0; k <= K; ++k) {
    destroyMat(Kmat[k]);
    destroyMat(Pmat[k]);
    destroyVec(tmpCvec[k]);
    destroyDA(da[k]); 
  }//end k

  PetscLogEventEnd(cleanupEvent, 0, 0, 0, 0);

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}





