
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

void buildPmat(std::vector<unsigned long long int>& factorialsList,
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DA>& da, std::vector<MPI_Comm>& activeComms,
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, std::vector<std::vector<PetscInt> >& partZ,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<int> >& offsets,
    std::vector<std::vector<int> >& scanLz, std::vector<std::vector<int> >& scanLy, std::vector<std::vector<int> >& scanLx, bool print);

void computePmat(std::vector<unsigned long long int>& factorialsList,
    Mat Pmat, int Nzc, int Nyc, int Nxc, int Nzf, int Nyf, int Nxf,
    std::vector<PetscInt>& lzc, std::vector<PetscInt>& lyc, std::vector<PetscInt>& lxc,
    std::vector<PetscInt>& lzf, std::vector<PetscInt>& lyf, std::vector<PetscInt>& lxf, 
    std::vector<int>& cOffsets, std::vector<int>& scanClz, std::vector<int>& scanCly, std::vector<int>& scanClx,
    std::vector<int>& fOffsets, std::vector<int>& scanFlz, std::vector<int>& scanFly, std::vector<int>& scanFlx,
    int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K); 

void buildKmat(std::vector<unsigned long long int>& factorialsList,
    std::vector<Mat>& Kmat, std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets, bool print);

void computeKmat(std::vector<unsigned long long int>& factorialsList,
    Mat Kmat, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<long long int>& coeffs, const unsigned int K, bool print);

void dirichletMatrixCorrection(Mat Kmat, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<int>& offsets);

void computeRandomRHS(DA da, Mat Kmat, Vec rhs, const unsigned int seed);

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms, int dim, int dofsPerNode, bool print);

void zeroBoundaries(DA da, Vec vec);

void createDA(std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dofsPerNode,
    int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, 
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<int> >& offsets,
    std::vector<std::vector<int> >& scanLz, std::vector<std::vector<int> >& scanLy,
    std::vector<std::vector<int> >& scanLx, MPI_Comm globalComm, bool print);

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt> & lz, std::vector<PetscInt> & ly, std::vector<PetscInt> & lx,
    std::vector<int>& offsets, std::vector<int>& scanLz, std::vector<int>& scanLy, std::vector<int>& scanLx);

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, bool print);

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void destroyComms(std::vector<MPI_Comm> & activeComms);

void destroyMat(std::vector<Mat> & mat);

void destroyVec(std::vector<Vec>& vec);

void destroyDA(std::vector<DA>& da); 

void destroyKSP(std::vector<KSP>& ksp);

#endif



