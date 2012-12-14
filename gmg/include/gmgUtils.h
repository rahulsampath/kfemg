
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

#define __MMS_X_PARAM__ 1
#define __MMS_Y_PARAM__ 1
#define __MMS_Z_PARAM__ 1

#define __SOLUTION_1D__(x) (sin(__MMS_X_PARAM__ * __PI__ * (x)))
#define __SOLUTION_2D__(x, y) (sin(__MMS_X_PARAM__ * __PI__ * (x)) * sin(__MMS_Y_PARAM__ * __PI__ * (y)))
#define __SOLUTION_3D__(x, y, z) (sin(__MMS_X_PARAM__ * __PI__ * (x)) * sin(__MMS_Y_PARAM__ * __PI__ * (y)) \
    * sin(__MMS_Z_PARAM__ * __PI__ * (z)))

#define __FORCE_1D__(x) ((__MMS_X_PARAM__ * __MMS_X_PARAM__) * (__PI__ * __PI__) * (__SOLUTION_1D__((x)))) 
#define __FORCE_2D__(x, y) (((__MMS_X_PARAM__ * __MMS_X_PARAM__) + (__MMS_Y_PARAM__ * __MMS_Y_PARAM__)) * \
    (__PI__ * __PI__) * (__SOLUTION_2D__((x), (y)))) 
#define __FORCE_3D__(x, y, z) (((__MMS_X_PARAM__ * __MMS_X_PARAM__) + (__MMS_Y_PARAM__ * __MMS_Y_PARAM__) + \
      (__MMS_Z_PARAM__ * __MMS_Z_PARAM__)) * (__PI__ * __PI__) * (__SOLUTION_3D__((x), (y), (z)))) 


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

/*
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
   Mat A;
   Mat B;
   Mat C;
   Vec aIn;
   Vec aOut;
   Vec bIn;
   Vec cOut;
   };

   struct SmatData {
   Mat A;
   Mat B;
   KSP cKsp;
   Vec aOut;
   Vec cRhs;
   Vec cSol;
   };

   struct SchurPCdata {
   Mat B;
   KSP cKsp;
   KSP sKsp;
   Vec cRhs;
   Vec x;
   Vec z;
   Vec sRhs;
   Vec sSol;
   };
   */

PetscErrorCode applyMG(PC pc, Vec in, Vec out);

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<KSP>& ksp, std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void coarseGridSolve(KSP ksp, Vec rhs, Vec sol); 

void smooth(KSP ksp, Vec rhs, Vec sol);

void createElementMatrices(std::vector<unsigned long long int>& factorialsList, int dim, int K, 
    std::vector<long long int>& coeffs, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<std::vector<long double> > >& elemMats, bool print);

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res);

void applyRestriction(Mat Pmat, Vec tmpCvec, Vec fVec, Vec cVec);

void applyProlongation(Mat Pmat, Vec tmpCvec, Vec cVec, Vec fVec);

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

void buildKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats, bool print);

void computeKmat(Mat Kmat, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, std::vector<long long int>& coeffs,
    const unsigned int K, bool print);

void dirichletMatrixCorrection(Mat Kmat, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets);

void computeRHS(DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<long long int>& coeffs, const int K, Vec rhs);

double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K);

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

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
    int dim, int dofsPerNode, bool print);

void destroyComms(std::vector<MPI_Comm> & activeComms);

void destroyMat(std::vector<Mat> & mat);

void destroyVec(std::vector<Vec>& vec);

void destroyDA(std::vector<DM>& da); 

void destroyKSP(std::vector<KSP>& ksp);

/*
   void createAllSchurPC(std::vector<std::vector<SchurPCdata> >& pcData, std::vector<std::vector<Mat> >& SmatShells,
   std::vector<Mat>& KmatShells, std::vector<std::vector<KmatData> >& kMatData,
   std::vector<std::vector<SmatData> >& sMatData);

   void createSchurPCdata(int lev, std::vector<SchurPCdata>& pcData, std::vector<Mat>& SmatShell, Mat& KmatShell, 
   std::vector<KmatData>& kMatData, std::vector<SmatData>& sMatData);

   void createSmatData(std::vector<Mat>& SmatShell, std::vector<SmatData>& data,
   std::vector<Mat>& KblkDiag, std::vector<Mat>& KblkUpper);

   void createAllSmatShells(std::vector<std::vector<Mat> >& SmatShells, std::vector<std::vector<SmatData> >& sMatData,
   std::vector<std::vector<Mat> >& KblkDiag, std::vector<std::vector<Mat> >& KblkUpper);

   void createKmatData(int blkId, Mat& KmatShell, std::vector<KmatData>& data,
   std::vector<Mat>& KblkDiag, std::vector<Mat>& KblkUpper);

   void createAllKmatShells(std::vector<Mat>& KmatShells, std::vector<std::vector<KmatData> >& kMatData,
   std::vector<std::vector<Mat> >& KblkDiag, std::vector<std::vector<Mat> >& KblkUpper);

   PetscErrorCode applySmatvec(Mat Smat, Vec in, Vec out);

   PetscErrorCode applyKmatvec(Mat Kmat, Vec in, Vec out);

   PetscErrorCode applySchurPC(PC pc, Vec in, Vec out);

   PetscErrorCode applyBlockPC(PC pc, Vec in, Vec out);

   void createBlockPCdata(std::vector<BlockPCdata>& data, std::vector<std::vector<Mat> >& KblkDiag,
   std::vector<std::vector<Mat> >& KblkUpper, bool print);

   void buildKdiagBlocks(std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
   std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
   std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
   std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats);

   void buildKupperBlocks(std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
   std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
   std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
   std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats);

   void computeKblkDiag(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
   std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, std::vector<long long int>& coeffs, 
   const unsigned int K, const unsigned int dof);

   void computeKblkUpper(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
   std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, std::vector<long long int>& coeffs, 
   const unsigned int K, const unsigned int dof);

   void dirichletMatrixCorrectionBlkDiag(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
   std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets);

   void dirichletMatrixCorrectionBlkUpper(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
   std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets);

   void computeRandomRHS(DM da, Mat Kmat, Vec rhs, const unsigned int seed);

   void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
   std::vector<BlockPCdata>& data, int dim, int dofsPerNode, bool print);

   void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
   std::vector<std::vector<SchurPCdata> >& data, int dim, int dofsPerNode, bool print);

   void destroyBlockPCdata(std::vector<BlockPCdata>& data);

   void destroyKmatData(std::vector<KmatData>& data);

   void destroySmatData(std::vector<SmatData>& data);

   void destroySchurPCdata(std::vector<SchurPCdata>& data);
   */

#endif



