
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
#include "common/include/commonUtils.h"

#include "gmg/include/exactSolution.i"

inline long double force1D(long double x) {
  long double res = -solutionDerivative1D(x, 2);
  return res;
}

inline long double force2D(long double x, long double y) {
  long double res = -( solutionDerivative2D(x, y, 2, 0) + solutionDerivative2D(x, y, 0, 2) );
  return res;
}

inline long double force3D(long double x, long double y, long double z) {
  long double res = -( solutionDerivative3D(x, y, z, 2, 0, 0) + 
      solutionDerivative3D(x, y, z, 0, 2, 0) + 
      solutionDerivative3D(x, y, z, 0, 0, 2) );
  return res;
}

struct MGdata {
  PetscInt numVcycles;
  std::vector<Mat> Kmat;
  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec; 
  std::vector<KSP> smoother;
  KSP coarseSolver;
  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;
};

struct PCFD1Ddata {
  KSP ksp;
  Vec rhs;
  Vec sol;
  Vec u;
  Vec uPrime;
  std::vector<PetscInt>* partX;
  int numDofs;
};

struct Khat1Ddata {
  Mat K11;
  Mat K12;
  Vec u;
  Vec uPrime;
  Vec tmpOut;
  std::vector<PetscInt>* partX;
  int numDofs;
};

struct Kcol1Ddata {
  std::vector<Mat> Kblk;
  Vec tmp;
  int nx;
};

PetscErrorCode applyPCFD1D(PC pc, Vec in, Vec out);

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res);

PetscErrorCode applyMG(PC pc, Vec in, Vec out);

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, 
    std::vector<Vec>& tmpCvec, std::vector<KSP>& smoother, KSP coarseSolver,
    std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void applyRestriction(Mat Pmat, Vec tmpCvec, Vec fVec, Vec cVec);

void applyProlongation(Mat Pmat, Vec tmpCvec, Vec cVec, Vec fVec);

void applyFD1D(MPI_Comm comm, std::vector<PetscInt>& partX, Vec in, Vec out);

PetscErrorCode Khat1Dmult(Mat mat, Vec in, Vec out);

PetscErrorCode Kcol1Dmult(Mat mat, Vec in, Vec out);

void buildPmat(int dim, PetscInt dofsPerNode, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes);

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

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

void buildBlkKmats(std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<DM>& da,
    std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes);

void assembleBlkKmats(std::vector<std::vector<std::vector<Mat> > >& blkKmats, int dim, int dofsPerNode,
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<DM>& da, int K, 
    std::vector<long long int>& coeffs, std::vector<unsigned long long int>& factorialsList);

void correctBlkKmats(int dim, std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<DM>& da,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, 
    std::vector<std::vector<PetscInt> >& offsets, int K);

void blkDirichletMatCorrection1D(std::vector<std::vector<Mat> >& blkKmat, DM da,
    std::vector<PetscInt>& offsets, int K);

void blkDirichletMatCorrection2D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partX,
    std::vector<PetscInt>& offsets, int K);

void blkDirichletMatCorrection3D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K);

void computeBlkKmat1D(Mat blkKmat, DM da, std::vector<PetscInt>& offsets,
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof);

void computeBlkKmat2D(Mat blkKmat, DM da, std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, 
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof);

void computeBlkKmat3D(Mat blkKmat, DM da, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, 
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof);

void buildKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, bool print);

void assembleKmat(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<Mat>& Kmat, std::vector<DM>& da, int K, std::vector<long long int>& coeffs,
    std::vector<unsigned long long int>& factorialsList, bool print);

void computeKmat(Mat Kmat, DM da, std::vector<std::vector<long double> >& elemMat);

void correctKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, int K);

void dirichletMatrixCorrection(Mat Kmat, DM da, const int K);

void createDA(int dim, int dofsPerNode, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny,
    std::vector<PetscInt>& Nx, std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY, 
    std::vector<std::vector<PetscInt> >& partX, std::vector<int>& activeNpes, std::vector<MPI_Comm>& activeComms,
    std::vector<DM>& da);

void createActiveComms(std::vector<int>& activeNpes, std::vector<MPI_Comm>& activeComms);

void createGrids(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny,
    std::vector<PetscInt>& Nx, bool print);

void createGrids1D(std::vector<PetscInt>& Nx, bool print);

void createGrids2D(std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, bool print);

void createGrids3D(std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, bool print);

void computePartition(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny,
    std::vector<PetscInt>& Nx, std::vector<std::vector<PetscInt> >& partZ, 
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, 
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<PetscInt> >& scanZ,
    std::vector<std::vector<PetscInt> >& scanY, std::vector<std::vector<PetscInt> >& scanX,
    std::vector<int>& activeNpes, bool print);

void computePartition3D(PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt>& partZ, std::vector<PetscInt>& partY, std::vector<PetscInt>& partX,
    std::vector<PetscInt>& offsets, std::vector<PetscInt>& scanZ, 
    std::vector<PetscInt>& scanY, std::vector<PetscInt>& scanX);

void computePartition2D(PetscInt Ny, PetscInt Nx, int maxNpes, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, std::vector<PetscInt>& scanY, 
    std::vector<PetscInt>& scanX);

void computePartition1D(PetscInt Nx, int maxNpes, std::vector<PetscInt>& partX, 
    std::vector<PetscInt>& offsets, std::vector<PetscInt>& scanX);

void computeRHS(DM da, std::vector<long long int>& coeffs, const int K, Vec rhs);

long double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K);

void setSolution(DM da, Vec vec, const int K);

void setBoundaries(DM da, Vec vec, const int K);

void destroyDA(std::vector<DM>& da); 

void destroyComms(std::vector<MPI_Comm>& activeComms);

void destroyMat(std::vector<Mat>& mat);

void destroyVec(std::vector<Vec>& vec);

void destroyKSP(std::vector<KSP>& ksp);

void destroyKhat1Dmat(Mat& mat);

void destroyKcol1Dmat(Mat& mat);

void destroyPCFD1Ddata(PCFD1Ddata* data); 

#endif



