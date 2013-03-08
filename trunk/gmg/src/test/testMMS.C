
#include "petsc.h"
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/mesh.h"
#include "gmg/include/mms.h"
#include "gmg/include/rtgPC.h"
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
  PetscInt K = 0;
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

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList); 

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
#ifdef DEBUG
  assert(activeNpes[activeNpes.size() - 1] == npes);
#endif

  std::vector<MPI_Comm> activeComms;
  createActiveComms(activeNpes, activeComms);

  std::vector<DM> daCK;
  createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, daCK);

  PetscLogEventEnd(meshEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildPmatEvent, 0, 0, 0, 0);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(dim, dofsPerNode, Pmat, tmpCvec, daCK, activeComms, activeNpes); 

  computePmat(dim, factorialsList, Pmat, Nz, Ny, Nx, partZ, partY, partX, offsets,
      scanZ, scanY, scanX, dofsPerNode, coeffsCK, K);

  PetscLogEventEnd(buildPmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  std::vector<Mat> Kmat;
  buildKmat(Kmat, daCK, print);

  assembleKmat(dim, Nz, Ny, Nx, Kmat, daCK, K, coeffsCK, factorialsList, print);

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(rhsEvent, 0, 0, 0, 0);

  Vec rhs;
  DMCreateGlobalVector(daCK[daCK.size() - 1], &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  computeRHS(daCK[daCK.size() - 1], coeffsCK, K, rhs);

  VecZeroEntries(sol);
  setBoundaries(daCK[daCK.size() - 1], sol, K);
  VecScale(sol, -1.0);
  MatMultAdd(Kmat[Kmat.size() - 1], sol, rhs, rhs);
  VecScale(sol, -1.0);
  makeBoundariesConsistent(daCK[daCK.size() - 1], sol, rhs, K);

  PetscLogEventEnd(rhsEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  correctKmat(Kmat, daCK, K);

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(solverSetupEvent, 0, 0, 0, 0);

  int nlevels = activeComms.size();
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
    setupRTG(pc, K, (nlevels - 1), daCK, Kmat, Pmat, tmpCvec);
  }
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetOperators(ksp, Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_PRECONDITIONER);
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

  long double err = computeError(daCK[daCK.size() - 1], sol, coeffsCK, K);

  PetscLogEventEnd(errEvent, 0, 0, 0, 0);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  PetscLogEventBegin(cleanupEvent, 0, 0, 0, 0);

  VecDestroy(&rhs);
  VecDestroy(&sol);
  KSPDestroy(&ksp);
  destroyMat(Kmat);
  destroyMat(Pmat);
  destroyVec(tmpCvec);
  destroyDA(daCK); 

  PetscLogEventEnd(cleanupEvent, 0, 0, 0, 0);

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}





