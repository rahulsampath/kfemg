
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscksp.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"

PetscCookie gmgCookie;
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

int main(int argc, char *argv[]) {
  PetscInitialize(&argc, &argv, "optionsTestGMG", PETSC_NULL);

  PetscCookieRegister("GMG", &gmgCookie);
  PetscLogEventRegister("DA", gmgCookie, &createDAevent);
  PetscLogEventRegister("Pmat", gmgCookie, &buildPmatEvent);
  PetscLogEventRegister("Pmem", gmgCookie, &PmemEvent);
  PetscLogEventRegister("fillP", gmgCookie, &fillPmatEvent);
  PetscLogEventRegister("Kmat", gmgCookie, &buildKmatEvent);
  PetscLogEventRegister("Kmem", gmgCookie, &KmemEvent);
  PetscLogEventRegister("ElemKmat", gmgCookie, &elemKmatEvent);
  PetscLogEventRegister("fillK", gmgCookie, &fillKmatEvent);
  PetscLogEventRegister("DMC", gmgCookie, &dirichletMatCorrectionEvent);
  PetscLogEventRegister("Vcycle", gmgCookie, &vCycleEvent);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);
  PetscInt K;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);
  PetscTruth useRandomRHS = PETSC_TRUE;
  PetscOptionsGetTruth(PETSC_NULL, "-useRandomRHS", &useRandomRHS, PETSC_NULL);
  PetscInt maxVcycles = 1;
  PetscOptionsGetInt(PETSC_NULL, "-maxVcycles", &maxVcycles, PETSC_NULL);
  PetscReal rTol = 1.0e-12;
  PetscOptionsGetReal(PETSC_NULL, "-rTol", &rTol, PETSC_NULL);
  PetscReal aTol = 1.0e-12;
  PetscOptionsGetReal(PETSC_NULL, "-aTol", &aTol, PETSC_NULL);

  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

  bool print = (globalRank == 0);

  int dofsPerNode = getDofsPerNode(dim, K);

  if(print) {
    std::cout<<"Dim = "<<dim<<std::endl;
    std::cout<<"K = "<<K<<std::endl;
    std::cout<<"DofsPerNode = "<<dofsPerNode<<std::endl;
    std::cout<<"Random-RHS = "<<useRandomRHS<<std::endl;
    std::cout<<"sizeof(double) = "<<(sizeof(double))<<std::endl;
    std::cout<<"sizeof(long double) = "<<(sizeof(long double))<<std::endl;
    std::cout<<"sizeof(PetscScalar) = "<<(sizeof(PetscScalar))<<std::endl;
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

  std::vector<Mat> Kmat;
  buildKmat(Kmat, da, coeffs, K, print);

  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
  buildPmat(Pmat, tmpCvec, da, activeComms, activeNpes, dim, dofsPerNode, coeffs, K, Nz, Ny, Nx, partZ, partY, partX, print);

  std::vector<KSP> ksp;
  createKSP(ksp, Kmat, activeComms, dim, dofsPerNode, print);

  Vec rhs;
  DACreateGlobalVector(da[da.size() - 1], &rhs);

  const unsigned int seed = (0x3456782  + (54763*globalRank));
  computeRandomRHS(da[da.size() - 1], Kmat[Kmat.size() - 1], rhs, seed);

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

  VecZeroEntries(sol);

  computeResidual(Kmat[Kmat.size() - 1], sol, rhs, res);
  PetscReal initialResNorm;
  VecNorm(res, NORM_2, &initialResNorm);
  if(print) {
    std::cout<<"Initial Residual Norm = "<<std::setprecision(15)<<initialResNorm<<std::endl;
  }

  int iter = 0;
  while(iter < maxVcycles) {
    applyVcycle((Kmat.size() - 1), Kmat, Pmat, tmpCvec, ksp, mgSol, mgRhs, mgRes);
    ++iter;
    computeResidual(Kmat[Kmat.size() - 1], sol, rhs, res);
    PetscReal currResNorm;
    VecNorm(res, NORM_2, &currResNorm);
    if(print) {
      std::cout<<"Iter = "<<iter<<" ResNorm = "<<std::setprecision(15)<<currResNorm<<std::endl;
    }
    if(currResNorm < aTol) {
      if(print) {
        std::cout<<"Converged due to ATOL."<<std::endl;
      }
      break;
    }
    if(currResNorm < (rTol*initialResNorm)) {
      if(print) {
        std::cout<<"Converged due to RTOL."<<std::endl;
      }
      break;
    }
  }//end while
  if(print) {
    std::cout<<"Number of V-cycles = "<<iter<<std::endl;
  }

  mgSol[Kmat.size() - 1] = NULL;
  mgRhs[Kmat.size() - 1] = NULL;
  mgRes[Kmat.size() - 1] = NULL;

  destroyVec(mgSol);
  destroyVec(mgRhs);
  destroyVec(mgRes);

  destroyKSP(ksp);

  VecDestroy(rhs);
  VecDestroy(res);
  VecDestroy(sol);

  destroyDA(da);

  destroyVec(tmpCvec);

  destroyMat(Pmat);

  destroyMat(Kmat);

  destroyComms(activeComms);

  PetscFinalize();

  return 0;
}


