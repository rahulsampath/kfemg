
#ifndef __GMG_UTILS__
#define __GMG_UTILS__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include "petscpc.h"
#include <vector>
#include "mpi.h"

struct MGdata {
  PetscInt numVcycles;
  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;
  std::vector<Mat> Kmat;
  std::vector<KSP> ksp;
  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec;
};

struct BlockPCdata {
  PetscInt numBlkIters;
  std::vector<Mat> KblkDiag;
  std::vector<Mat> KblkUpper;
  std::vector<KSP> blkKsp;
  Vec diagIn;
  Vec diagOut;
  std::vector<Vec> upperIn;
};

struct KmatData {
};

struct SmatData {
};

struct SchurPCdata {
};

PetscErrorCode applyMG(PC pc, Vec in, Vec out);

PetscErrorCode applyShellPC(PC pc, Vec in, Vec out);

void createElementMatrices(std::vector<unsigned long long int>& factorialsList, int dim, int K, 
    std::vector<long long int>& coeffs, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<std::vector<long double> > >& elemMats, bool print);

void createBlockPCdata(std::vector<BlockPCdata>& data, std::vector<std::vector<Mat> >& KblkDiag,
    std::vector<std::vector<Mat> >& KblkUpper, bool print);

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<KSP>& ksp, std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void applyRestriction(Mat Pmat, Vec tmpCvec, Vec fVec, Vec cVec);

void applyProlongation(Mat Pmat, Vec tmpCvec, Vec cVec, Vec fVec);

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res);

void buildPmat(std::vector<unsigned long long int>& factorialsList,
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms,
    std::vector<int>& activeNpes, int dim, PetscInt dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, 
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY, 
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<PetscInt> >& offsets,
    std::vector<std::vector<PetscInt> >& scanLz, std::vector<std::vector<PetscInt> >& scanLy,
    std::vector<std::vector<PetscInt> >& scanLx);

void computePmat(std::vector<unsigned long long int>& factorialsList,
    Mat Pmat, PetscInt Nzc, PetscInt Nyc, PetscInt Nxc, PetscInt Nzf, PetscInt Nyf, PetscInt Nxf,
    std::vector<PetscInt>& lzc, std::vector<PetscInt>& lyc, std::vector<PetscInt>& lxc,
    std::vector<PetscInt>& lzf, std::vector<PetscInt>& lyf, std::vector<PetscInt>& lxf, 
    std::vector<PetscInt>& cOffsets, std::vector<PetscInt>& scanClz,
    std::vector<PetscInt>& scanCly, std::vector<PetscInt>& scanClx,
    std::vector<PetscInt>& fOffsets, std::vector<PetscInt>& scanFlz, 
    std::vector<PetscInt>& scanFly, std::vector<PetscInt>& scanFlx,
    int dim, PetscInt dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K); 

void buildKmat(std::vector<unsigned long long int>& factorialsList,
    std::vector<Mat>& Kmat, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats, bool print);

void buildKdiagBlocks(std::vector<unsigned long long int>& factorialsList,
    std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats);

void buildKupperBlocks(std::vector<unsigned long long int>& factorialsList,
    std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats);

void computeKblkDiag(std::vector<unsigned long long int>& factorialsList,
    Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, std::vector<long long int>& coeffs, 
    const unsigned int K, const unsigned int dof);

void computeKblkUpper(std::vector<unsigned long long int>& factorialsList,
    Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, std::vector<long long int>& coeffs, 
    const unsigned int K, const unsigned int dof);

void computeKmat(std::vector<unsigned long long int>& factorialsList,
    Mat Kmat, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, std::vector<long long int>& coeffs,
    const unsigned int K, bool print);

void dirichletMatrixCorrection(Mat Kmat, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets);

void dirichletMatrixCorrectionBlkDiag(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets);

void dirichletMatrixCorrectionBlkUpper(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets);

void computeRandomRHS(DM da, Mat Kmat, Vec rhs, const unsigned int seed);

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
    std::vector<BlockPCdata>& data, int dim, int dofsPerNode, bool print);

void zeroBoundaries(DM da, Vec vec);

void createDA(std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dofsPerNode,
    int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, 
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<PetscInt> >& offsets,
    std::vector<std::vector<PetscInt> >& scanLz, std::vector<std::vector<PetscInt> >& scanLy,
    std::vector<std::vector<PetscInt> >& scanLx, MPI_Comm globalComm, bool print);

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt> & lz, std::vector<PetscInt> & ly, std::vector<PetscInt> & lx,
    std::vector<PetscInt>& offsets, std::vector<PetscInt>& scanLz, 
    std::vector<PetscInt>& scanLy, std::vector<PetscInt>& scanLx);

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, bool print);

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void destroyBlockPCdata(std::vector<BlockPCdata>& data);

void destroyComms(std::vector<MPI_Comm> & activeComms);

void destroyMat(std::vector<Mat> & mat);

void destroyVec(std::vector<Vec>& vec);

void destroyDA(std::vector<DM>& da); 

void destroyKSP(std::vector<KSP>& ksp);

#endif



