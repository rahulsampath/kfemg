
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include "mpi.h"
#include "petsc.h"
#include "petscksp.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"

void applyFDapproxType2(std::vector<double>& in, std::vector<double>& out) {
  int nx = in.size();

  std::vector<double> refine1(1 + 2*(nx - 1));
  std::vector<double> refine2(1 + 4*(nx - 1));
  std::vector<double> refine3(1 + 8*(nx - 1));

  for(int i = 0; i < nx; ++i) {
    refine1[2*i] = in[i];
  }//end i
  for(int i = 0; i < (nx - 1); ++i) {
    refine1[(2*i) + 1] = 0.5*(in[i] + in[i + 1]);
  }//end i

  int sz = refine1.size();
  for(int i = 0; i < sz; ++i) {
    refine2[2*i] = refine1[i];
  }//end i
  for(int i = 0; i < (sz - 1); ++i) {
    refine2[(2*i) + 1] = 0.5*(refine1[i] + refine1[i + 1]);
  }//end i

  sz = refine2.size();
  for(int i = 0; i < sz; ++i) {
    refine3[2*i] = refine2[i];
  }//end i
  for(int i = 0; i < (sz - 1); ++i) {
    refine3[(2*i) + 1] = 0.5*(refine2[i] + refine2[i + 1]);
  }//end i

  int len = refine3.size();
  std::vector<double> tmp(len);

  /*
  //First Order
  for(int i = 0; i < (len - 1); ++i) {
  tmp[i] = 2.0*(refine2[i + 1] - refine2[i]);
  }//end i
  tmp[len - 1] = 2.0*(refine2[len - 1] - refine2[len - 2]);
  */

  //Fourth Order
  tmp[0] = -((25.0 * refine2[0]) - (48.0 * refine2[1]) + (36.0 * refine2[2]) - (16.0 * refine2[3]) + (3.0 * refine2[4]))/3.0;
  tmp[1] = -((25.0 * refine2[1]) - (48.0 * refine2[2]) + (36.0 * refine2[3]) - (16.0 * refine2[4]) + (3.0 * refine2[5]))/3.0;
  for(int i = 2; i < (len - 2); ++i) {
    tmp[i] = (-refine2[i + 2] + (8.0 * refine2[i + 1]) - (8.0 * refine2[i - 1]) + refine2[i - 2])/3.0;
  }//end i
  tmp[len - 2] = ((25.0 * refine2[len - 2]) - (48.0 * refine2[len - 3]) + (36.0 * refine2[len - 4])
      - (16.0 * refine2[len - 5]) + (3.0 * refine2[len - 6]))/3.0;
  tmp[len - 1] = ((25.0 * refine2[len - 1]) - (48.0 * refine2[len - 2]) + (36.0 * refine2[len - 3])
      - (16.0 * refine2[len - 4]) + (3.0 * refine2[len - 5]))/3.0;

  for(int i = 0; i < nx; ++i) {
    out[i] = tmp[8*i];
  }//end i
}

void applyFDapproxType1(std::vector<double>& in, std::vector<double>& out) {
  int nx = in.size();

  /*
  //First Order
  for(int i = 0; i < (nx - 1); ++i) {
  out[i] = (in[i + 1] - in[i])/2.0;
  }//end i
  out[nx - 1] = (in[nx - 1] - in[nx - 2])/2.0;

  //Second Order
  out[0] = -((3.0 * in[0]) - (4.0 * in[1]) + in[2])/4.0;
  for(int i = 1; i < (nx - 1); ++i) {
  out[i] = (in[i + 1] - in[i - 1])/4.0;
  }//end i
  out[nx - 1] = ((3.0 * in[nx - 1]) - (4.0 * in[nx - 2]) + in[nx - 3])/4.0;
  */

  //Fourth Order
  out[0] = -((25.0 * in[0]) - (48.0 * in[1]) + (36.0 * in[2]) - (16.0 * in[3]) + (3.0 * in[4]))/24.0;
  out[1] = -((25.0 * in[1]) - (48.0 * in[2]) + (36.0 * in[3]) - (16.0 * in[4]) + (3.0 * in[5]))/24.0;
  for(int i = 2; i < (nx - 2); ++i) {
    out[i] = (-in[i + 2] + (8.0 * in[i + 1]) - (8.0 * in[i - 1]) + in[i - 2])/24.0;
  }//end i
  out[nx - 2] = ((25.0 * in[nx - 2]) - (48.0 * in[nx - 3]) + (36.0 * in[nx - 4]) - (16.0 * in[nx - 5]) + (3.0 * in[nx - 6]))/24.0;
  out[nx - 1] = ((25.0 * in[nx - 1]) - (48.0 * in[nx - 2]) + (36.0 * in[nx - 3]) - (16.0 * in[nx - 4]) + (3.0 * in[nx - 5]))/24.0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsTest1", PETSC_NULL);

  PetscInt dim = 1; 
  PetscInt K = 1;

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

  std::vector<Mat> Kmat;
  buildKmat(Kmat, da, print);

  assembleKmat(dim, Nz, Ny, Nx, Kmat, da, K, coeffs, factorialsList, print);

  correctKmat(Kmat, da, K);

  int nlevels = da.size();

  Vec sol;
  Vec rhs;
  MatGetVecs(Kmat[nlevels - 1], &sol, &rhs);

  PetscInt mode = 1;
  PetscOptionsGetInt(PETSC_NULL, "-mode", &mode, PETSC_NULL);

  PetscInt modeDof = 2;
  PetscOptionsGetInt(PETSC_NULL, "-modeDof", &modeDof, PETSC_NULL);

  PetscInt type = 1;
  PetscOptionsGetInt(PETSC_NULL, "-type", &type, PETSC_NULL);

  double hx = 1.0/(static_cast<double>(Nx[nlevels - 1] - 1));

  VecZeroEntries(sol);

  //  VecSetRandom(sol, PETSC_NULL);
  double* solArr;
  VecGetArray(sol, &solArr);
  for(int i = 0; i < Nx[nlevels - 1]; ++i) {
    if((modeDof == 0) || (modeDof == 2)) {
      solArr[(2*i)] = sin((static_cast<double>(mode*i))*__PI__*hx);
    }
    if((modeDof == 1) || (modeDof == 2)) {
      solArr[(2*i) + 1] = 0.5*hx*(static_cast<double>(mode))*(__PI__)*cos((static_cast<double>(mode*i))*__PI__*hx);
    }
  }//end i
  VecRestoreArray(sol, &solArr);
  MatMult(Kmat[nlevels - 1], sol, rhs);

  KSP ksp;
  PC pc;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);
  KSPSetPCSide(ksp, PC_LEFT);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCCHOLESKY);
  KSPSetOperators(ksp, Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_PRECONDITIONER);
  KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
  KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 1);
  KSPSetOptionsPrefix(ksp, "plain_");
  KSPSetFromOptions(ksp);

  KSPSolve(ksp, rhs, sol);

  Vec res;
  VecDuplicate(rhs, &res);

  computeResidual(Kmat[nlevels - 1], sol, rhs, res);

  PetscReal norm;
  VecNorm(res, NORM_2, &norm);
  std::cout<<"True Res = "<<norm<<std::endl;

  double* resArr;
  VecGetArray(res, &resArr);

  /*
     std::cout<<"Dof-0:"<<std::endl;
     for(int i = 0; i < Nx[nlevels - 1]; ++i) {
     std::cout<<"res["<<i<<"] = "<<(resArr[(2*i)])<<std::endl;
     }//end i

     std::cout<<"Dof-1:"<<std::endl;
     for(int i = 0; i < Nx[nlevels - 1]; ++i) {
     std::cout<<"res["<<i<<"] = "<<(resArr[(2*i) + 1])<<std::endl;
     }//end i
     */

  VecRestoreArray(res, &resArr);

  VecGetArray(sol, &solArr);

  std::vector<double> uVals(Nx[nlevels - 1]);
  std::vector<double> uPrimeVals(Nx[nlevels - 1]);
  for(int i = 0; i < Nx[nlevels - 1]; ++i) {
    uVals[i] = solArr[2*i];
  }//end i
  if(type == 1) {
    applyFDapproxType1(uVals, uPrimeVals);
  } else {
    applyFDapproxType2(uVals, uPrimeVals);
  }
  for(int i = 0; i < Nx[nlevels - 1]; ++i) {
    std::cout<<"Sol["<<i<<"] = "<<(solArr[(2*i)])<<" SolPrime = "<<(solArr[(2*i) + 1])<<" Est = "<<(uPrimeVals[i])<<std::endl;
    std::cout<<"Exact = "<<(sin((static_cast<double>(mode*i))*__PI__*hx))<<" Prime = "<<
      (0.5*hx*(static_cast<double>(mode))*(__PI__)*cos((static_cast<double>(mode*i))*__PI__*hx))<<std::endl;
  }//end i
  for(int i = 0; i < Nx[nlevels - 1]; ++i) {
    solArr[(2*i) + 1] = uPrimeVals[i];
  }//end i
  VecRestoreArray(sol, &solArr);

  computeResidual(Kmat[nlevels - 1], sol, rhs, res);

  VecNorm(res, NORM_2, &norm);
  std::cout<<"Res using Est = "<<norm<<std::endl;

  VecGetArray(res, &resArr);

  /*
     std::cout<<"Dof-0:"<<std::endl;
     for(int i = 0; i < Nx[nlevels - 1]; ++i) {
     std::cout<<"res["<<i<<"] = "<<(resArr[(2*i)])<<std::endl;
     }//end i

     std::cout<<"Dof-1:"<<std::endl;
     for(int i = 0; i < Nx[nlevels - 1]; ++i) {
     std::cout<<"res["<<i<<"] = "<<(resArr[(2*i) + 1])<<std::endl;
     }//end i
     */

  VecRestoreArray(res, &resArr);

  VecDestroy(&res);
  VecDestroy(&sol);
  VecDestroy(&rhs);
  destroyMat(Kmat);
  destroyDA(da); 

  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}







