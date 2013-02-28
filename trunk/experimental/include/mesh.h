
#ifndef __MESH__
#define __MESH__

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

void destroyDA(std::vector<DM>& da); 

void destroyComms(std::vector<MPI_Comm>& activeComms);

#endif


