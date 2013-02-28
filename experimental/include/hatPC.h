
#ifndef __HAT_PC__
#define __HAT_PC__

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
  Vec u;
  Vec uPrime;
  Vec tmpOut;
  std::vector<std::vector<Mat> >* blkKmats;
  std::vector<PetscInt>* partX;
  int numDofs;
  int K;
};

void createAll1DmatShells(int K, std::vector<MPI_Comm>& activeComms, 
    std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<Mat> >& Khat1Dmats);

void create1DmatShells(MPI_Comm comm, int K, std::vector<std::vector<Mat> >& blkKmats,
    std::vector<PetscInt>& partX, std::vector<Mat>& Khat1Dmats);

PetscErrorCode Khat1Dmult(Mat mat, Vec in, Vec out);

void createAll1DhatPc(std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<std::vector<Mat> > >& blkKmats,
    std::vector<std::vector<Mat> >& Khat1Dmats, std::vector<std::vector<PC> >& hatPc);

PetscErrorCode applyPCFD1D(PC pc, Vec in, Vec out);

void buildBlkKmats(std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<DM>& da,
    std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes);

void assembleBlkKmats(std::vector<std::vector<std::vector<Mat> > >& blkKmats, int dim, int dofsPerNode,
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<DM>& da, int K, 
    std::vector<long long int>& coeffs, std::vector<unsigned long long int>& factorialsList);

void correctBlkKmats(int dim, std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<DM>& da,
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<PetscInt> >& offsets, int K);

void blkDirichletMatCorrection1D(std::vector<std::vector<Mat> >& blkKmat, DM da,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K);

void blkDirichletMatCorrection2D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K);

void blkDirichletMatCorrection3D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partZ,
    std::vector<PetscInt>& partY, std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K);

void computeBlkKmat1D(Mat blkKmat, DM da, std::vector<PetscInt>& offsets,
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof);

void computeBlkKmat2D(Mat blkKmat, DM da, std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, 
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof);

void computeBlkKmat3D(Mat blkKmat, DM da, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, 
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof);

void destroyPCFD1Ddata(PCFD1Ddata* data); 

void destroyKhat1Ddata(Khat1Ddata* data);

#endif



