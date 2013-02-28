
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
#include "gmg/include/mms.h"
#include "gmg/include/assembly.h"
#include "gmg/include/boundary.h"
#include "gmg/include/intergrid.h"
#include "gmg/include/mgPC.h"
#include "gmg/include/lsFitType3PC.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsNewTestMMS", PETSC_NULL);

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

  std::vector<long long int> coeffsCK;
  read1DshapeFnCoeffs(K, coeffsCK);

  std::vector<long long int> coeffsC0;
  if(K > 0) {
    read1DshapeFnCoeffs(0, coeffsC0);
  }

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

  std::vector<DM> daCK;
  createDA(dim, dofsPerNode, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, daCK);

  std::vector<DM> daC0;
  if(K > 0) {
    createDA(dim, 1, Nz, Ny, Nx, partZ, partY, partX, activeNpes, activeComms, daC0);
  }

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(dim, dofsPerNode, Pmat, tmpCvec, daCK, activeComms, activeNpes); 

  computePmat(dim, factorialsList, Pmat, Nz, Ny, Nx, partZ, partY, partX, offsets,
      scanZ, scanY, scanX, dofsPerNode, coeffsCK, K);

  std::vector<Mat> Kmat;
  buildKmat(Kmat, daCK, print);

  std::vector<Mat> reducedMat;
  if(K > 0) {
    buildKmat(reducedMat, daC0, false);
  }

  assembleKmat(dim, Nz, Ny, Nx, Kmat, daCK, K, coeffsCK, factorialsList, print);

  if(K > 0) {
    assembleKmat(dim, Nz, Ny, Nx, reducedMat, daC0, 0, coeffsC0, factorialsList, false);
  }

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

  correctKmat(Kmat, daCK, K);

  if(K > 0) {
    correctKmat(reducedMat, daC0, 0);
  }

  std::vector<KSP> smoother(Pmat.size(), NULL);
  for(int lev = 0; lev < (smoother.size()); ++lev) {
    if(rank < activeNpes[lev + 1]) {
      KSPCreate(activeComms[lev + 1], &(smoother[lev]));
      if(K > 0) {
        KSPSetType(smoother[lev], KSPFGMRES);
      } else {
        KSPSetType(smoother[lev], KSPCG);
      }
      PC smoothPC;
      KSPGetPC(smoother[lev], &smoothPC);
      if(K == 0) {
        PCSetType(smoothPC, PCNONE);
      } else {
        PCSetType(smoothPC, PCSHELL);
        setupLSfitType3PC(smoothPC, Kmat[lev + 1], reducedMat[lev + 1],
            K, Nx[lev + 1], coeffsCK, coeffsC0);
      }
      KSPSetInitialGuessNonzero(smoother[lev], PETSC_TRUE);
      KSPSetOperators(smoother[lev], Kmat[lev + 1], Kmat[lev + 1], SAME_PRECONDITIONER);
      KSPSetTolerances(smoother[lev], 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetOptionsPrefix(smoother[lev], "smooth_");
      KSPSetFromOptions(smoother[lev]);
    }
  }//end lev

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

  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;
  buildMGworkVecs(Kmat, mgSol, mgRhs, mgRes);

  MGdata data;
  data.K = K;
  data.daFinest = daCK[daCK.size() - 1];
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
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 10);
  KSPSetOptionsPrefix(ksp, "outer_");
  KSPSetFromOptions(ksp);

  if(print) {
    std::cout<<"Solving..."<<std::endl;
  }

  KSPSolve(ksp, rhs, sol);

  long double err = computeError(daCK[daCK.size() - 1], sol, coeffsCK, K);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  VecDestroy(&rhs);
  VecDestroy(&sol);

  if(coarseSolver != NULL) {
    KSPDestroy(&coarseSolver);
  }
  destroyKSP(smoother);
  KSPDestroy(&ksp);

  destroyMat(Pmat);
  destroyVec(tmpCvec);
  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);
  destroyMat(Kmat);
  destroyMat(reducedMat);
  destroyDA(daCK); 
  destroyDA(daC0); 

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}




