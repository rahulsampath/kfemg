
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include "mpi.h"
#include "petsc.h"
#include "petscksp.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsTestMMS", PETSC_NULL);

  PetscInt dim = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-dim", &dim, PETSC_NULL);
  assert(dim > 0);
  assert(dim <= 3);
  PetscInt K;
  PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int npes;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

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

  //Compute Partition
  PetscInt Nx = 5;
  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &Nx, PETSC_NULL);

  PetscInt Ny = 1;
  if(dim > 1) {
    PetscOptionsGetInt(PETSC_NULL, "-finestNy", &Ny, PETSC_NULL);
  }

  int px, py;
  if(dim == 1) {
    px = npes;
  } else if(dim == 2) {
    px = sqrt(npes);
    py = npes/px;
    assert((px*py) == npes);
  } else {
    assert(false);
  }

  assert(px >= 1);
  assert(px <= Nx);
  if(dim > 1) {
    assert(py >= 1);
    assert(py <= Ny);
  }

  std::vector<PetscInt> partX;
  PetscInt avgX = Nx/px;
  PetscInt extraX = Nx%px; 
  partX.resize(px, avgX);
  for(int cnt = 0; cnt < extraX; ++cnt) {
    ++(partX[cnt]);
  }//end cnt

  std::vector<PetscInt> partY;
  PetscInt avgY = Ny/py;
  PetscInt extraY = Ny%py; 
  partY.resize(py, avgY);
  for(int cnt = 0; cnt < extraY; ++cnt) {
    ++(partY[cnt]);
  }//end cnt

  //Create DA
  DM da;
  if(dim == 1) {
    DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, Nx, dofsPerNode, 1, &(partX[0]), &da);
  } else if(dim == 2) {
    DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX,
        Nx, Ny, px, py, dofsPerNode, 1, &(partX[0]), &(partY[0]), &da);
  } else {
    assert(false);
  }

  //Build Kmat
  Mat Kmat;
  DMCreateMatrix(da, MATAIJ, &Kmat);

  PetscInt sz;
  MatGetSize(Kmat, &sz, PETSC_NULL);
  if(print) {
    std::cout<<"Kmat Size = "<<sz<<std::endl;
  }

  std::vector<std::vector<long double> > elemMat;
  if(dim == 1) {
    long double hx = 1.0L/(static_cast<long double>(Nx - 1));
    createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, print);
  } else if(dim == 2) {
    long double hx = 1.0L/(static_cast<long double>(Nx - 1));
    long double hy = 1.0L/(static_cast<long double>(Ny - 1));
    createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, print);
  } else {
    assert(false);
  }

  Vec rhs;
  DMCreateGlobalVector(da, &rhs);

  Vec sol;
  VecDuplicate(rhs, &sol); 

  //ComputeRHS
  //computeRHS(da, coeffs, K, rhs);
  VecZeroEntries(rhs);

  //Neumann Matrix (Assembly)
  computeKmat(Kmat, da, elemMat);

  //ModifyRHS 
  VecZeroEntries(sol);
  setBoundaries(da, sol);
  VecScale(sol, -1.0);
  MatMultAdd(Kmat, sol, rhs, rhs);
  setBoundaries(da, rhs);

  //std::cout<<"Matrix before DirichletCorrection: "<<std::endl;
  //MatView(Kmat, PETSC_VIEWER_STDOUT_WORLD);

  dirichletMatrixCorrection(Kmat, da);

  //std::cout<<"Matrix after DirichletCorrection: "<<std::endl;
  //MatView(Kmat, PETSC_VIEWER_STDOUT_WORLD);

  //Build KSP
  PC pc;
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);
  KSPSetPCSide(ksp, PC_LEFT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCCHOLESKY);
  PCFactorSetShiftAmount(pc, 1.0e-13);
  PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
  KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
  KSPSetOperators(ksp, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 50);
  KSPSetFromOptions(ksp);

  KSPSolve(ksp, rhs, sol);

  //Compute Error
  long double err = computeError(da, sol, coeffs, K);

  if(print) {
    std::cout<<"Error = "<<std::setprecision(13)<<err<<std::endl;
  }

  chkBoundaries(da, sol);

  VecDestroy(&rhs);
  VecDestroy(&sol);

  KSPDestroy(&ksp);

  MatDestroy(&Kmat);

  DMDestroy(&da);

  PetscFinalize();

  MPI_Finalize();

  return 0;
}






