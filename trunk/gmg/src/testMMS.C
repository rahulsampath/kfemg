
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscksp.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"

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

  PetscInitialize(&argc, &argv, "optionsTestMMS", PETSC_NULL);

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
  assert(activeNpes[activeNpes.size() - 1] == npes);

  std::vector<MPI_Comm> activeComms;
  createActiveComms(activeNpes, activeComms);

  std::vector<DM> da;
  createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, da);

  PetscLogEventEnd(meshEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildPmatEvent, 0, 0, 0, 0);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(dim, dofsPerNode, Pmat, tmpCvec, da, activeComms, activeNpes); 

  computePmat(dim, factorialsList, Pmat, Nz, Ny, Nx, partZ, partY, partX, offsets,
      scanZ, scanY, scanX, dofsPerNode, coeffs, K);

  PetscLogEventEnd(buildPmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  //Build Kmat
  std::vector<std::vector<std::vector<Mat> > > blkKmats;
  buildBlkKmats(blkKmats, da, activeComms, activeNpes);

  std::vector<Mat> Kmat;
  buildKmat(Kmat, da, print);

  //Matrix Assembly
  assembleKmat(dim, Nz, Ny, Nx, Kmat, da, K, coeffs, factorialsList, print);

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(rhsEvent, 0, 0, 0, 0);

  Vec rhs;
  DMCreateGlobalVector(da[da.size() - 1], &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  //ComputeRHS
  computeRHS(da[da.size() - 1], coeffs, K, rhs);

  //Boundary corrections to rhs and sol. 
  VecZeroEntries(sol);
  setBoundaries(da[da.size() - 1], sol, K);
  VecScale(sol, -1.0);
  MatMultAdd(Kmat[Kmat.size() - 1], sol, rhs, rhs);
  setBoundaries(da[da.size() - 1], rhs, K);
  VecScale(sol, -1.0);

  PetscLogEventEnd(rhsEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  correctKmat(Kmat, da, K);

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(solverSetupEvent, 0, 0, 0, 0);

  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;
  buildMGworkVecs(Kmat, mgSol, mgRhs, mgRes);

  KSP coarseSolver = NULL;
  if(rank < activeNpes[0]) {
    PC coarsePC;
    KSPCreate(activeComms[0], &coarseSolver);
    KSPSetType(coarseSolver, KSPCG);
    KSPSetPCSide(coarseSolver, PC_LEFT);
    KSPGetPC(coarseSolver, &coarsePC);
    PCSetType(coarsePC, PCCHOLESKY);
    PCFactorSetShiftAmount(coarsePC, 1.0e-12);
    PCFactorSetShiftType(coarsePC, MAT_SHIFT_POSITIVE_DEFINITE);
    PCFactorSetMatSolverPackage(coarsePC, MATSOLVERMUMPS);
    KSPSetInitialGuessNonzero(coarseSolver, PETSC_TRUE);
    KSPSetOperators(coarseSolver, Kmat[0], Kmat[0], SAME_PRECONDITIONER);
    KSPSetTolerances(coarseSolver, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
    KSPSetOptionsPrefix(coarseSolver, "coarse_");
    KSPSetFromOptions(coarseSolver);
  }

  std::vector<KSP> smoother(Pmat.size(), NULL);
  for(int lev = 0; lev < (smoother.size()); ++lev) {
    if(rank < activeNpes[lev + 1]) {
      PC smoothPC;
      KSPCreate(activeComms[lev + 1], &(smoother[lev]));
      KSPSetType(smoother[lev], KSPFGMRES);
      KSPSetPCSide(smoother[lev], PC_RIGHT);
      KSPGetPC(smoother[lev], &smoothPC);
      PCSetType(smoothPC, PCNONE);
      KSPSetInitialGuessNonzero(smoother[lev], PETSC_TRUE);
      KSPSetOperators(smoother[lev], Kmat[lev + 1], Kmat[lev + 1], SAME_PRECONDITIONER);
      KSPSetTolerances(smoother[lev], 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetOptionsPrefix(smoother[lev], "smooth_");
      KSPSetFromOptions(smoother[lev]);
    }
  }//end lev

  MGdata data;
  PetscInt numVcycles = 1;
  PetscOptionsGetInt(PETSC_NULL, "-numVcycles", &numVcycles, PETSC_NULL);
  if(print) {
    std::cout<<"numVcycles = "<<numVcycles<<std::endl;
  }
  data.numVcycles = numVcycles;
  data.Kmat = Kmat;
  data.Pmat = Pmat;
  data.tmpCvec = tmpCvec; 
  data.smoother = smoother;
  data.coarseSolver = coarseSolver;
  data.mgSol = mgSol;
  data.mgRhs = mgRhs;
  data.mgRes = mgRes;

  //Build KSP
  KSP ksp;
  PC pc;
  KSPCreate(activeComms[activeComms.size() - 1], &ksp);
  KSPSetType(ksp, KSPFGMRES);
  KSPSetPCSide(ksp, PC_RIGHT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCSHELL);
  PCShellSetContext(pc, &data);
  PCShellSetName(pc, "MyVcycle");
  PCShellSetApply(pc, &applyMG);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  KSPSetOperators(ksp, Kmat[Kmat.size() - 1], Kmat[Kmat.size() - 1], SAME_PRECONDITIONER);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
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

  //Compute Error
  long double err = computeError(da[da.size() - 1], sol, coeffs, K);

  PetscLogEventEnd(errEvent, 0, 0, 0, 0);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  PetscLogEventBegin(cleanupEvent, 0, 0, 0, 0);

  VecDestroy(&rhs);
  VecDestroy(&sol);

  KSPDestroy(&ksp);
  if(coarseSolver != NULL) {
    KSPDestroy(&coarseSolver);
  }
  destroyKSP(smoother);

  destroyMat(Pmat);
  destroyVec(tmpCvec);
  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);
  destroyMat(Kmat);
  destroyDA(da); 

  for(int i = 0; i < blkKmats.size(); ++i) {
    for(int j = 0; j < blkKmats[i].size(); ++j) {
      destroyMat(blkKmats[i][j]);
    }//end j
  }//end i

  PetscLogEventEnd(cleanupEvent, 0, 0, 0, 0);

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}









