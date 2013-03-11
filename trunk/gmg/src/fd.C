
#include "gmg/include/fd.h"
#include <vector>

#ifdef DEBUG
#include <cassert>
#endif

void applyFD(DM da, int K, int px, int py, int pz, Vec in, Vec out) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)da), &comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  int npes;
  MPI_Comm_size(comm, &npes);

  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DMDAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

#ifdef DEBUG
  assert(Nx > 2);
  if(dim > 1) {
    assert(Ny > 2);
  }
  if(dim > 2) {
    assert(Nz > 2);
  }
#endif

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  long double hy = 0;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz = 0;
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  }

  if(dim == 1) {
    PetscScalar** inArr;
    PetscScalar** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);
    double L1, L2, R1, R2;
    MPI_Request sReq1;
    MPI_Request rReq1;
    MPI_Request sReq2;
    MPI_Request rReq2;
    if(rank > 0) {
      MPI_Irecv(&L1, 1, MPI_DOUBLE, (rank - 1), 2, comm, &rReq2);
      MPI_Isend(&(inArr[xs][K - 1]), 1, MPI_DOUBLE, (rank - 1), 1, comm, &sReq1);
    }
    if(rank < (npes - 1)) {
      MPI_Irecv(&R1, 1, MPI_DOUBLE, (rank + 1), 1, comm, &rReq1);
      MPI_Isend(&(inArr[xs + nx - 1][K - 1]), 1, MPI_DOUBLE, (rank + 1), 2, comm, &sReq2);
    }
    for(int xi = xs + 1; xi < (xs + nx - 1); ++xi) {
      outArr[xi][K] = (inArr[xi + 1][K - 1] - inArr[xi - 1][K - 1])/4.0;
    }//end xi
    MPI_Status status;
    if(rank > 0) {
      MPI_Wait(&sReq1, &status);
      MPI_Wait(&rReq2, &status);
    }
    if(rank < (npes - 1)) {
      MPI_Wait(&sReq2, &status);
      MPI_Wait(&rReq1, &status);
    }
    MPI_Request sReq3;
    MPI_Request rReq3;
    MPI_Request sReq4;
    MPI_Request rReq4;
    if((xs == 0) && (nx == 1)) {
      MPI_Irecv(&R2, 1, MPI_DOUBLE, (rank + 1), 3, comm, &rReq3);
    }
    if(xs == 1) {
      if(nx == 1) {
        MPI_Isend(&R1, 1, MPI_DOUBLE, (rank - 1), 3, comm, &sReq3);
      } else {
        MPI_Isend(&(inArr[xs + 1][K - 1]), 1, MPI_DOUBLE, (rank - 1), 3, comm, &sReq3);
      }
    }
    if(((xs + nx) == Nx) && (nx == 1)) {
      MPI_Irecv(&L2, 1, MPI_DOUBLE, (rank - 1), 4, comm, &rReq4);
    }
    if((xs + nx) == (Nx - 1)) {
      if(nx == 1) {
        MPI_Isend(&L1, 1, MPI_DOUBLE, (rank + 1), 4, comm, &sReq4);
      } else {
        MPI_Isend(&(inArr[xs + nx - 2][K - 1]), 1, MPI_DOUBLE, (rank + 1), 4, comm, &sReq4);
      }
    }
    if((xs == 0) && (nx == 1)) {
      MPI_Wait(&rReq3, &status);
    }
    if(xs == 1) {
      MPI_Wait(&sReq3, &status);
    }
    if(((xs + nx) == Nx) && (nx == 1)) {
      MPI_Wait(&rReq4, &status);
    }
    if((xs + nx) == (Nx - 1)) {
      MPI_Wait(&sReq4, &status);
    }
    if(rank == 0) {
      if(nx == 1) {
        outArr[xs][K] = -((3.0*inArr[xs][K - 1]) - (4.0*R1) + R2)/4.0;
      } else if(nx == 2) {
        outArr[xs][K] = -((3.0*inArr[xs][K - 1]) - (4.0*inArr[xs + 1][K - 1]) + R1)/4.0;
        outArr[xs + 1][K] = (R1 - inArr[xs][K - 1])/4.0;
      } else {
        outArr[xs][K] = -((3.0*inArr[xs][K - 1]) - (4.0*inArr[xs + 1][K - 1]) + inArr[xs + 2][K - 1])/4.0;
        if(npes == 1) {
          outArr[xs + nx - 1][K] = ((3.0*inArr[xs + nx - 1][K - 1]) - (4.0*inArr[xs + nx - 2][K - 1]) + inArr[xs + nx - 3][K - 1])/4.0;
        } else {
          outArr[xs + nx - 1][K] = (R1 - inArr[xs + nx - 2][K - 1])/4.0;
        }
      }
    } else if(rank == (npes - 1)) {
      if(nx == 1) {
        outArr[xs][K] = ((3.0*inArr[xs][K - 1]) - (4.0*L1) + L2)/4.0;
      } else if(nx == 2) {
        outArr[xs][K] = (inArr[xs + 1][K - 1] - L1)/4.0;
        outArr[xs + 1][K] = ((3.0*inArr[xs + 1][K - 1]) - (4.0*inArr[xs][K - 1]) + L1)/4.0;
      } else {
        outArr[xs][K] = (inArr[xs + 1][K - 1] - L1)/4.0;
        outArr[xs + nx - 1][K] = ((3.0*inArr[xs + nx - 1][K - 1]) - (4.0*inArr[xs + nx - 2][K - 1]) + inArr[xs + nx - 3][K - 1])/4.0;
      }
    } else {
      if(nx == 1) {
        outArr[xs][K] = (R1 - L1)/4.0;
      } else {
        outArr[xs][K] = (inArr[xs + 1][K - 1] - L1)/4.0;
        outArr[xs + nx - 1][K] = (R1 - inArr[xs + nx - 2][K - 1])/4.0;
      }
    }
    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  } else if(dim == 2) {
    int rj = rank/px;
    int ri = rank%px;
    PetscScalar*** inArr;
    PetscScalar*** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);
    std::vector<double> L1(ny);
    std::vector<double> L2(ny);
    std::vector<double> R1(ny);
    std::vector<double> R2(ny);
    std::vector<double> F1(nx);
    std::vector<double> F2(nx);
    std::vector<double> H1(nx);
    std::vector<double> H2(nx);
    std::vector<double> firstX(ny);
    std::vector<double> lastX(ny);
    std::vector<double> firstY(nx);
    std::vector<double> lastY(nx);
    int prevX = (rj*px) + (ri - 1);
    int nextX = (rj*px) + (ri + 1);
    int prevY = ((rj - 1)*px) + ri;
    int nextY = ((rj + 1)*px) + ri;

    //dx
    {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        //firstX[yi - ys] = inArr[yi][xs][];
      }//end yi
    }

    //dy
    {
    }

    //dxdy
    {
    }

    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  } else {
    int rk = rank/(px*py);
    int rj = (rank/px)%py;
    int ri = rank%px;
    PetscScalar**** inArr;
    PetscScalar**** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);

    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  }
}


