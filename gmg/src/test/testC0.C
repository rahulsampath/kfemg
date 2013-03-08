
#include "petsc.h"
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/mesh.h"
#include "gmg/include/mms.h"
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

void applyVcycle(int currLev, std::vector<KSP>& ksp, std::vector<Mat>& Kmat,
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<Vec>& rhs,
    std::vector<Vec>& sol, std::vector<Vec>& res) {
  KSPSolve(ksp[currLev], rhs[currLev], sol[currLev]);
  if(currLev > 0) {
    computeResidual(Kmat[currLev], sol[currLev], rhs[currLev], res[currLev]);
    applyRestriction(Pmat[currLev - 1], tmpCvec[currLev - 1], res[currLev], rhs[currLev - 1]);
    if(sol[currLev - 1] != NULL) {
      VecZeroEntries(sol[currLev - 1]);
      applyVcycle((currLev - 1), ksp, Kmat, Pmat, tmpCvec, rhs, sol, res);
    }
    applyProlongation(Pmat[currLev - 1], tmpCvec[currLev - 1], sol[currLev - 1], res[currLev]);
    VecAXPY(sol[currLev], 1.0, res[currLev]);
    KSPSolve(ksp[currLev], rhs[currLev], sol[currLev]);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsC0", PETSC_NULL);

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
#ifdef DEBUG
  assert(activeNpes[activeNpes.size() - 1] == npes);
#endif

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

  std::vector<Mat> Kmat;
  buildKmat(Kmat, da, print);

  assembleKmat(dim, Nz, Ny, Nx, Kmat, da, K, coeffs, factorialsList, print);

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(rhsEvent, 0, 0, 0, 0);

  Vec rhs;
  DMCreateGlobalVector(da[da.size() - 1], &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  computeRHS(da[da.size() - 1], coeffs, K, rhs);

  VecZeroEntries(sol);
  setBoundaries(da[da.size() - 1], sol, K);
  VecScale(sol, -1.0);
  MatMultAdd(Kmat[Kmat.size() - 1], sol, rhs, rhs);
  VecScale(sol, -1.0);
  makeBoundariesConsistent(da[da.size() - 1], sol, rhs, K);

  PetscLogEventEnd(rhsEvent, 0, 0, 0, 0);

  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  correctKmat(Kmat, da, K);

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);

  PetscLogEventBegin(solverSetupEvent, 0, 0, 0, 0);

  int nlevels = activeComms.size();
  std::vector<KSP> ksp(nlevels, NULL);
  if(activeComms[0] != MPI_COMM_NULL) {
    PC pc;
    KSPCreate(activeComms[0], &(ksp[0]));
    KSPGetPC(ksp[0], &pc);
    KSPSetType(ksp[0], KSPPREONLY);
    KSPSetPCSide(ksp[0], PC_LEFT);
    PCSetType(pc, PCCHOLESKY);
    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
    KSPSetInitialGuessNonzero(ksp[0], PETSC_FALSE);
    KSPSetOperators(ksp[0], Kmat[0], Kmat[0], SAME_PRECONDITIONER);
    KSPSetTolerances(ksp[0], 1.0e-12, 1.0e-12, 2.0, 1);
  }
  for(int i = 1; i < nlevels; ++i) {
    if(activeComms[i] != MPI_COMM_NULL) {
      PC pc;
      KSPCreate(activeComms[i], &(ksp[i]));
      KSPGetPC(ksp[i], &pc);
      KSPSetType(ksp[i], KSPRICHARDSON);
      if(dim == 1) {
        KSPRichardsonSetScale(ksp[i], (2.0/3.0));
      } else if(dim == 2) {
        KSPRichardsonSetScale(ksp[i], (4.0/5.0));
      } else {
        KSPRichardsonSetScale(ksp[i], (8.0/9.0));
      }
      KSPSetPCSide(ksp[i], PC_LEFT);
      PCSetType(pc, PCJACOBI);
      KSPSetInitialGuessNonzero(ksp[i], PETSC_TRUE);
      KSPSetOperators(ksp[i], Kmat[i], Kmat[i], SAME_PRECONDITIONER);
      KSPSetTolerances(ksp[i], 1.0e-12, 1.0e-12, 2.0, 2);
    }
  }//end i

  std::vector<Vec> res(nlevels, NULL);
  for(int i = 1; i < nlevels; ++i) {
    if(Kmat[i] != NULL) {
      MatGetVecs(Kmat[i], PETSC_NULL, &(res[i]));
    }
  }//end i

  std::vector<Vec> tmpRhs(nlevels, NULL);
  std::vector<Vec> tmpSol(nlevels, NULL);
  for(int i = 0; i < (nlevels - 1); ++i) {
    if(Kmat[i] != NULL) {
      MatGetVecs(Kmat[i], &(tmpSol[i]), &(tmpRhs[i]));
    }
  }//end i
  tmpRhs[nlevels - 1] = rhs;
  tmpSol[nlevels - 1] = sol;

  PetscLogEventEnd(solverSetupEvent, 0, 0, 0, 0);

  if(print) {
    std::cout<<"Solving..."<<std::endl;
  }

  PetscLogEventBegin(solverApplyEvent, 0, 0, 0, 0);

  computeResidual(Kmat[nlevels - 1], sol, rhs, res[nlevels - 1]);
  PetscReal currNorm;
  VecNorm(res[nlevels - 1], NORM_2, &currNorm);
  PetscReal initNorm = currNorm;
  for(int iter = 0; iter < 50; ++iter) {
    if(print) {
      std::cout<<"Iter = "<<iter<<" Res = "<<currNorm<<std::endl;
    }
    if(currNorm <= 1.0e-12) {
      break;
    }
    if(currNorm <= (1.0e-12*initNorm)) {
      break;
    }
    applyVcycle((nlevels - 1), ksp, Kmat, Pmat, tmpCvec, tmpRhs, tmpSol, res);
    computeResidual(Kmat[nlevels - 1], sol, rhs, res[nlevels - 1]);
    VecNorm(res[nlevels - 1], NORM_2, &currNorm);
  }//end iter

  PetscLogEventEnd(solverApplyEvent, 0, 0, 0, 0);

  PetscLogEventBegin(errEvent, 0, 0, 0, 0);

  long double err = computeError(da[da.size() - 1], sol, coeffs, K);

  PetscLogEventEnd(errEvent, 0, 0, 0, 0);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  PetscLogEventBegin(cleanupEvent, 0, 0, 0, 0);

  destroyMat(Kmat);
  destroyMat(Pmat);
  destroyKSP(ksp);
  destroyVec(tmpCvec);
  destroyVec(res);
  destroyVec(tmpRhs);
  destroyVec(tmpSol);
  destroyDA(da); 

  PetscLogEventEnd(cleanupEvent, 0, 0, 0, 0);

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}



