
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

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res);

void buildPmat(std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DA>& da,
    std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dim, int dofsPerNode);

void buildKmat(std::vector<Mat>& Kmat, std::vector<DA>& da);

void computeRandomRHS(DA da, Mat Kmat, Vec rhs, const unsigned int seed);

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms);

void zeroBoundaries(DA da, Vec vec);

void createDA(std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dofsPerNode,
    int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, MPI_Comm globalComm);

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes, int &pz, int &py, int &px);

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx);

#endif



