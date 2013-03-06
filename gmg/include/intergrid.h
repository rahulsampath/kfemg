
#ifndef __INTERGRID__
#define __INTERGRID__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "mpi.h"
#include <vector>

void applyRestriction(Mat Pmat, Vec tmpCvec, Vec fVec, Vec cVec);

void applyProlongation(Mat Pmat, Vec tmpCvec, Vec cVec, Vec fVec);

void buildPmat(int dim, PetscInt dofsPerNode, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes);

void computePmat(int dim, std::vector<unsigned long long int>& factorialsList, std::vector<Mat>& Pmat, 
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, 
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<PetscInt> >& offsets,
    std::vector<std::vector<PetscInt> >& scanZ, std::vector<std::vector<PetscInt> >& scanY,
    std::vector<std::vector<PetscInt> >& scanX, PetscInt dofsPerNode,
    std::vector<long long int>& coeffs, const int K);

void computePmat1D(std::vector<unsigned long long int>& factorialsList, Mat Pmat,
    PetscInt Nxc, PetscInt Nxf, std::vector<PetscInt>& partXc, std::vector<PetscInt>& partXf,
    std::vector<PetscInt>& cOffsets, std::vector<PetscInt>& scanXc,
    std::vector<PetscInt>& fOffsets, std::vector<PetscInt>& scanXf,
    PetscInt dofsPerNode, std::vector<long long int>& coeffs, const int K); 

void computePmat2D(std::vector<unsigned long long int>& factorialsList,
    Mat Pmat, PetscInt Nyc, PetscInt Nxc, PetscInt Nyf, PetscInt Nxf,
    std::vector<PetscInt>& partYc, std::vector<PetscInt>& partXc,
    std::vector<PetscInt>& partYf, std::vector<PetscInt>& partXf, 
    std::vector<PetscInt>& cOffsets, std::vector<PetscInt>& scanYc, std::vector<PetscInt>& scanXc,
    std::vector<PetscInt>& fOffsets, std::vector<PetscInt>& scanYf, std::vector<PetscInt>& scanXf,
    PetscInt dofsPerNode, std::vector<long long int>& coeffs, const int K); 

void computePmat3D(std::vector<unsigned long long int>& factorialsList,
    Mat Pmat, PetscInt Nzc, PetscInt Nyc, PetscInt Nxc, PetscInt Nzf, PetscInt Nyf, PetscInt Nxf,
    std::vector<PetscInt>& partZc, std::vector<PetscInt>& partYc, std::vector<PetscInt>& partXc,
    std::vector<PetscInt>& partZf, std::vector<PetscInt>& partYf, std::vector<PetscInt>& partXf, 
    std::vector<PetscInt>& cOffsets, std::vector<PetscInt>& scanZc,
    std::vector<PetscInt>& scanYc, std::vector<PetscInt>& scanXc,
    std::vector<PetscInt>& fOffsets, std::vector<PetscInt>& scanZf, 
    std::vector<PetscInt>& scanYf, std::vector<PetscInt>& scanXf,
    PetscInt dofsPerNode, std::vector<long long int>& coeffs, const int K); 

#endif


