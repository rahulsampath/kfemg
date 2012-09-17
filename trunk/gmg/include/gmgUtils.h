
#ifndef __GMG_UTILS__
#define __GMG_UTILS__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscda.h"
#include "petscksp.h"
#include "petscpc.h"
#include <vector>
#include "mpi.h"

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<KSP>& ksp, std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void applyRestriction(Mat Pmat, Vec tmpCvec, Vec fVec, Vec cVec);

void applyProlongation(Mat Pmat, Vec tmpCvec, Vec cVec, Vec fVec);

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res);

void buildPmat(std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DA>& da,
    std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dim, int dofsPerNode,
    std::vector<long long int>& coeffs, const unsigned int K);

void computePmat(Mat Pmat, DA dac, DA daf, std::vector<long long int>& coeffs, const unsigned int K, MPI_Comm comm); 

void buildKmat(std::vector<Mat>& Kmat, std::vector<DA>& da, std::vector<long long int>& coeffs, const unsigned int K);

void computeKmat(Mat Kmat, DA da, std::vector<long long int>& coeffs, const unsigned int K);

void dirichletMatrixCorrection(Mat Kmat, DA da);

void computeRandomRHS(DA da, Mat Kmat, Vec rhs, const unsigned int seed);

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms);

void zeroBoundaries(DA da, Vec vec);

void createDA(std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dofsPerNode,
    int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, MPI_Comm globalComm);

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt> & lz, std::vector<PetscInt> & ly, std::vector<PetscInt> & lx);

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx);

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void destroyComms(std::vector<MPI_Comm> & activeComms);

void destroyMat(std::vector<Mat> & mat);

void destroyVec(std::vector<Vec>& vec);

void destroyDA(std::vector<DA>& da); 

void destroyKSP(std::vector<KSP>& ksp);

#endif



