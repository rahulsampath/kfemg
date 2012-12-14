
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
PetscLogEvent errEvent;
PetscLogEvent rhsEvent;

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
  PetscLogEventRegister("Error", gmgCookie, &errEvent);
  PetscLogEventRegister("RHS", gmgCookie, &rhsEvent);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);
  PetscInt K;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);

  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

  bool print = (globalRank == 0);

  int dofsPerNode = getDofsPerNode(dim, K);

  if(print) {
    std::cout<<"Dim = "<<dim<<std::endl;
    std::cout<<"K = "<<K<<std::endl;
    std::cout<<"DofsPerNode = "<<dofsPerNode<<std::endl;
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
  buildKmat(Kmat, da, activeComms, activeNpes, coeffs, K,
      partZ, partY, partX, offsets, elemMats, print);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(factorialsList, Pmat, tmpCvec, da, activeComms, activeNpes, dim, dofsPerNode, coeffs, K, Nz, Ny, Nx,
      partZ, partY, partX, offsets, scanLz, scanLy, scanLx);

  if(print) {
    std::cout<<"Built P matrices."<<std::endl;
  }

  /*
     std::vector<std::vector<Mat> > KblkDiag;
     buildKdiagBlocks(KblkDiag, da, activeComms, activeNpes, coeffs, K, 
     partZ, partY, partX, offsets, elemMats);

     if(print) {
     std::cout<<"Built K-diag blocks."<<std::endl;
     }

     std::vector<std::vector<Mat> > KblkUpper;
     buildKupperBlocks(KblkUpper, da, activeComms, activeNpes, coeffs, K,
     partZ, partY, partX, offsets, elemMats);

     if(print) {
     std::cout<<"Built K-upper blocks."<<std::endl;
     }

     std::vector<Mat> KmatShells;
     std::vector<std::vector<KmatData> > kMatData;
     createAllKmatShells(KmatShells, kMatData, KblkDiag, KblkUpper);

     if(print) {
     std::cout<<"Built Kmat Shells."<<std::endl;
     }

     std::vector<std::vector<Mat> > SmatShells;
     std::vector<std::vector<SmatData> > sMatData;
     createAllSmatShells(SmatShells, sMatData, KblkDiag, KblkUpper);

     if(print) {
     std::cout<<"Built Smat Shells."<<std::endl;
     }

     std::vector<std::vector<SchurPCdata> > schurPCdata;
     createAllSchurPC(schurPCdata, SmatShells, KmatShells, kMatData, sMatData);

     if(print) {
     std::cout<<"Built SchurPC."<<std::endl;
     }

     std::vector<KSP> ksp;
     createKSP(ksp, Kmat, activeComms, schurPCdata, dim, dofsPerNode, print);
     */

  /*
     std::vector<BlockPCdata> blkPCdata;
     createBlockPCdata(blkPCdata, KblkDiag, KblkUpper, print);

     if(print) {
     std::cout<<"Built BlockPC."<<std::endl;
     }

     std::vector<KSP> ksp;
     createKSP(ksp, Kmat, activeComms, blkPCdata, dim, dofsPerNode, print);
     */

  std::vector<KSP> ksp;
  createKSP(ksp, Kmat, activeComms, dim, dofsPerNode, print);

  Vec rhs;
  DMCreateGlobalVector(da[da.size() - 1], &rhs);

  computeRHS(da[da.size() - 1], partZ[da.size() - 1], partY[da.size() - 1],
      partX[da.size() - 1], offsets[da.size() - 1], coeffs, K, rhs);

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
  PCShellSetName(pc, "MyVcycle");
  PCShellSetApply(pc, &applyMG);
  KSPSetInitialGuessNonzero(outerKsp, PETSC_FALSE);
  KSPSetOperators(outerKsp, Kmat[Kmat.size() - 1], Kmat[Kmat.size() - 1], SAME_PRECONDITIONER);
  KSPSetTolerances(outerKsp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
  KSPSetOptionsPrefix(outerKsp, "outer_");
  KSPSetFromOptions(outerKsp);

  KSPSolve(outerKsp, rhs, sol);

  double err = computeError(da[da.size() - 1], sol, coeffs, K);

  if(print) {
    std::cout<<"Error = "<<err<<std::endl;
  }

  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);

  KSPDestroy(&outerKsp);

  destroyKSP(ksp);

  VecDestroy(&rhs);
  VecDestroy(&sol);

  /*
     destroyBlockPCdata(blkPCdata);

     for(size_t i = 0; i < KblkDiag.size(); ++i) {
     destroyMat(KblkDiag[i]);
     }//end i

     for(size_t i = 0; i < KblkUpper.size(); ++i) {
     destroyMat(KblkUpper[i]);
     }//end i
     */

  destroyMat(Kmat);

  /*
     for(size_t i = 0; i < schurPCdata.size(); ++i) {
     destroySchurPCdata(schurPCdata[i]);
     }//end i
     schurPCdata.clear();

     for(size_t i = 0; i < SmatShells.size(); ++i) {
     destroyMat(SmatShells[i]);
     }//end i
     SmatShells.clear();

     for(size_t i = 0; i < sMatData.size(); ++i) {
     destroySmatData(sMatData[i]);
     }//end i
     sMatData.clear();
     */

  /*
     for(size_t i = 0; i < KmatShells.size(); ++i) {
     if(KmatShells[i] != NULL) {
     PetscBool same;
     PetscObjectTypeCompare(((PetscObject)(KmatShells[i])), MATSHELL, &same);
     if(same) {
     MatDestroy(&(KmatShells[i]));
     }
     }
     }//end i
     KmatShells.clear();

     for(size_t i = 0; i < kMatData.size(); ++i) {
     destroyKmatData(kMatData[i]);
     }//end i
     kMatData.clear();
     */

  destroyDA(da);

  destroyVec(tmpCvec);

  destroyMat(Pmat);

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}


