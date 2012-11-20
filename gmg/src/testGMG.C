
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
PetscLogEvent createDAevent;
PetscLogEvent buildPmatEvent;
PetscLogEvent buildKmatEvent;
PetscLogEvent buildKblkDiagEvent;
PetscLogEvent buildKblkUpperEvent;
PetscLogEvent vCycleEvent;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsTestGMG", PETSC_NULL);

  PetscClassIdRegister("GMG", &gmgCookie);
  PetscLogEventRegister("DA", gmgCookie, &createDAevent);
  PetscLogEventRegister("Pmat", gmgCookie, &buildPmatEvent);
  PetscLogEventRegister("Kmat", gmgCookie, &buildKmatEvent);
  PetscLogEventRegister("KblkD", gmgCookie, &buildKblkDiagEvent);
  PetscLogEventRegister("KblkU", gmgCookie, &buildKblkUpperEvent);
  PetscLogEventRegister("Vcycle", gmgCookie, &vCycleEvent);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);
  PetscInt K;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);
  PetscBool useRandomRHS = PETSC_TRUE;
  PetscOptionsGetBool(PETSC_NULL, "-useRandomRHS", &useRandomRHS, PETSC_NULL);

  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

  bool print = (globalRank == 0);

  int dofsPerNode = getDofsPerNode(dim, K);

  if(print) {
    std::cout<<"Dim = "<<dim<<std::endl;
    std::cout<<"K = "<<K<<std::endl;
    std::cout<<"DofsPerNode = "<<dofsPerNode<<std::endl;
    std::cout<<"Random-RHS = "<<useRandomRHS<<std::endl;
  }

  std::vector<DM> da;
  std::vector<PetscInt> Nx;
  std::vector<PetscInt> Ny;
  std::vector<PetscInt> Nz;
  std::vector<MPI_Comm> activeComms;
  std::vector<int> activeNpes;
  std::vector<std::vector<PetscInt> > partZ;
  std::vector<std::vector<PetscInt> > partY;
  std::vector<std::vector<PetscInt> > partX;
  std::vector<std::vector<PetscInt> > offsets;
  std::vector<std::vector<PetscInt> > scanLz;
  std::vector<std::vector<PetscInt> > scanLy;
  std::vector<std::vector<PetscInt> > scanLx;
  createDA(da, activeComms, activeNpes, dofsPerNode, dim, Nz, Ny, Nx, partZ, partY, partX,
      offsets, scanLz, scanLy, scanLx, MPI_COMM_WORLD, print);

  assert(da[da.size() - 1] != NULL);

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList); 

  std::vector<std::vector<std::vector<long double> > > elemMats;
  createElementMatrices(factorialsList, dim, K, coeffs, Nz, Ny, Nx, elemMats, print);

  if(print) {
    std::cout<<"Created Element Matrices."<<std::endl;
  }

  std::vector<Mat> Kmat;
  buildKmat(factorialsList, Kmat, da, activeComms, activeNpes, dim, dofsPerNode, coeffs,
      K, partZ, partY, partX, offsets, elemMats, print);

  std::vector<std::vector<Mat> > KblkDiag;
  buildKdiagBlocks(factorialsList, KblkDiag, da, activeComms, activeNpes, coeffs, K, 
      partZ, partY, partX, offsets, elemMats);

  std::vector<std::vector<Mat> > KblkUpper;
  buildKupperBlocks(factorialsList, KblkUpper, da, activeComms, activeNpes, dim, dofsPerNode, coeffs,
      K, partZ, partY, partX, offsets, elemMats);

  std::vector<PCShellData> shellData;
  createPCShellData(shellData, KblkDiag, KblkUpper, print);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(factorialsList, Pmat, tmpCvec, da, activeComms, activeNpes, dim, dofsPerNode, coeffs, K, Nz, Ny, Nx,
      partZ, partY, partX, offsets, scanLz, scanLy, scanLx, print);

  std::vector<KSP> ksp;
  createKSP(ksp, Kmat, activeComms, shellData, dim, dofsPerNode, print);

  Vec rhs;
  DMCreateGlobalVector(da[da.size() - 1], &rhs);

  const unsigned int seed = (0x3456782  + (54763*globalRank));
  computeRandomRHS(da[da.size() - 1], Kmat[Kmat.size() - 1], rhs, seed);

  Vec sol;
  VecDuplicate(rhs, &sol);

  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;

  buildMGworkVecs(Kmat, mgSol, mgRhs, mgRes);

  MGdata data;

  PetscInt numVcycles = 1;
  PetscOptionsGetInt(PETSC_NULL, "-numVcycles", &numVcycles, PETSC_NULL);
  if(print) {
    std::cout<<"numVcycles = "<<numVcycles<<std::endl;
  }
  data.numVcycles = numVcycles;
  data.mgSol = mgSol;
  data.mgRhs = mgRhs;
  data.mgRes = mgRes;
  data.Kmat = Kmat;
  data.Pmat = Pmat;
  data.tmpCvec = tmpCvec;
  data.ksp = ksp;

  PC pc;
  KSP outerKsp;
  KSPCreate(PETSC_COMM_WORLD, &outerKsp);
  KSPGetPC(outerKsp, &pc);
  KSPSetType(outerKsp, KSPFGMRES);
  KSPSetPCSide(outerKsp, PC_RIGHT);
  PCSetType(pc, PCSHELL);
  PCShellSetContext(pc, &data);
  PCShellSetApply(pc, &applyMG);
  KSPSetInitialGuessNonzero(outerKsp, PETSC_FALSE);
  KSPSetOperators(outerKsp, Kmat[Kmat.size() - 1], Kmat[Kmat.size() - 1], SAME_PRECONDITIONER);
  KSPSetTolerances(outerKsp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 20);
  KSPSetOptionsPrefix(outerKsp, "outer_");
  KSPSetFromOptions(outerKsp);

  KSPSolve(outerKsp, rhs, sol);

  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);

  KSPDestroy(&outerKsp);

  destroyKSP(ksp);

  VecDestroy(&rhs);
  VecDestroy(&sol);

  destroyPCShellData(shellData);

  destroyDA(da);

  destroyVec(tmpCvec);

  destroyMat(Pmat);

  for(size_t i = 0; i < KblkDiag.size(); ++i) {
    destroyMat(KblkDiag[i]);
  }//end i

  for(size_t i = 0; i < KblkUpper.size(); ++i) {
    destroyMat(KblkUpper[i]);
  }//end i

  destroyMat(Kmat);

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}


