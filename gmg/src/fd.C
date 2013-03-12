
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
    if(rank < (px - 1)) {
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
    if(rank < (px - 1)) {
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
    if(((xs + nx) == Nx) && (nx == 1)) {
      MPI_Irecv(&L2, 1, MPI_DOUBLE, (rank - 1), 4, comm, &rReq4);
    }
    if(xs == 1) {
      if(nx == 1) {
        MPI_Isend(&R1, 1, MPI_DOUBLE, (rank - 1), 3, comm, &sReq3);
      } else {
        MPI_Isend(&(inArr[xs + 1][K - 1]), 1, MPI_DOUBLE, (rank - 1), 3, comm, &sReq3);
      }
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
    if(((xs + nx) == Nx) && (nx == 1)) {
      MPI_Wait(&rReq4, &status);
    }
    if(xs == 1) {
      MPI_Wait(&sReq3, &status);
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
        if(px == 1) {
          outArr[xs + nx - 1][K] = ((3.0*inArr[xs + nx - 1][K - 1]) - (4.0*inArr[xs + nx - 2][K - 1]) + inArr[xs + nx - 3][K - 1])/4.0;
        } else {
          outArr[xs + nx - 1][K] = (R1 - inArr[xs + nx - 2][K - 1])/4.0;
        }
      }
    } else if(rank == (px - 1)) {
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
    int prevX = (rj*px) + (ri - 1);
    int nextX = (rj*px) + (ri + 1);
    int prevY = ((rj - 1)*px) + ri;
    int nextY = ((rj + 1)*px) + ri;
    PetscScalar*** inArr;
    PetscScalar*** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);

    //dx
    {
      int len = ny*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int d = 0; d < K; ++d) {
          int dof = (d*(K + 1)) + K - 1;
          first[((yi - ys)*K) + d] = inArr[yi][xs][dof];
          last[((yi - ys)*K) + d] = inArr[yi][xs + nx - 1][dof];
        }//end d
      }//end yi
      MPI_Request sReq1;
      MPI_Request rReq1;
      MPI_Request sReq2;
      MPI_Request rReq2;
      if(ri > 0) {
        MPI_Irecv(&(L1[0]), len, MPI_DOUBLE, prevX, 2, comm, &rReq2);
        MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevX, 1, comm, &sReq1);
      }
      if(ri < (px - 1)) {
        MPI_Irecv(&(R1[0]), len, MPI_DOUBLE, nextX, 1, comm, &rReq1);
        MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextX, 2, comm, &sReq2);
      }
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int xi = xs + 1; xi < (xs + nx - 1); ++xi) {
          for(int d = 0; d < K; ++d) {
            int outDof = (d*(K + 1)) + K;
            int inDof = (d*(K + 1)) + K - 1;
            outArr[yi][xi][outDof] = (inArr[yi][xi + 1][inDof] - inArr[yi][xi - 1][inDof])/4.0;
          }//end d
        }//end xi
      }//end yi
      MPI_Status status;
      if(ri > 0) {
        MPI_Wait(&sReq1, &status);
        MPI_Wait(&rReq2, &status);
      }
      if(ri < (px - 1)) {
        MPI_Wait(&sReq2, &status);
        MPI_Wait(&rReq1, &status);
      }
      MPI_Request sReq3;
      MPI_Request rReq3;
      MPI_Request sReq4;
      MPI_Request rReq4;
      if((xs == 0) && (nx == 1)) {
        MPI_Irecv(&(R2[0]), len, MPI_DOUBLE, nextX, 3, comm, &rReq3);
      }
      if(((xs + nx) == Nx) && (nx == 1)) {
        MPI_Irecv(&(L2[0]), len, MPI_DOUBLE, prevX, 4, comm, &rReq4);
      }
      if(xs == 1) {
        if(nx == 1) {
          MPI_Isend(&(R1[0]), len, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        } else {
          for(int yi = ys; yi < (ys + ny); ++yi) {
            for(int d = 0; d < K; ++d) {
              int dof = (d*(K + 1)) + K - 1;
              first[((yi - ys)*K) + d] = inArr[yi][xs + 1][dof];
            }//end d
          }//end yi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        }
      }
      if((xs + nx) == (Nx - 1)) {
        if(nx == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        } else {
          for(int yi = ys; yi < (ys + ny); ++yi) {
            for(int d = 0; d < K; ++d) {
              int dof = (d*(K + 1)) + K - 1;
              last[((yi - ys)*K) + d] = inArr[yi][xs + nx - 2][dof];
            }//end d
          }//end yi
          MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        }
      }
      if((xs == 0) && (nx == 1)) {
        MPI_Wait(&rReq3, &status);
      }
      if(((xs + nx) == Nx) && (nx == 1)) {
        MPI_Wait(&rReq4, &status);
      }
      if(xs == 1) {
        MPI_Wait(&sReq3, &status);
      }
      if((xs + nx) == (Nx - 1)) {
        MPI_Wait(&sReq4, &status);
      }
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int d = 0; d < K; ++d) {
          int outDof = (d*(K + 1)) + K;
          int inDof = (d*(K + 1)) + K - 1;
          int gDof = ((yi - ys)*K) + d;
          if(ri == 0) {
            if(nx == 1) {
              outArr[yi][xs][outDof] = -((3.0*inArr[yi][xs][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
            } else if(nx == 2) {
              outArr[yi][xs][outDof] = -((3.0*inArr[yi][xs][inDof]) -
                  (4.0*inArr[yi][xs + 1][inDof]) + R1[gDof])/4.0;
              outArr[yi][xs + 1][outDof] = (R1[gDof] - inArr[yi][xs][inDof])/4.0;
            } else {
              outArr[yi][xs][outDof] = -((3.0*inArr[yi][xs][inDof]) - (4.0*inArr[yi][xs + 1][inDof])
                  + inArr[yi][xs + 2][inDof])/4.0;
              if(px == 1) {
                outArr[yi][xs + nx - 1][outDof] = ((3.0*inArr[yi][xs + nx - 1][inDof]) -
                    (4.0*inArr[yi][xs + nx - 2][inDof]) + inArr[yi][xs + nx - 3][inDof])/4.0;
              } else {
                outArr[yi][xs + nx - 1][outDof] = (R1[gDof] - inArr[yi][xs + nx - 2][inDof])/4.0;
              }
            }
          } else if(ri == (px - 1)) {
            if(nx == 1) {
              outArr[yi][xs][outDof] = ((3.0*inArr[yi][xs][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
            } else if(nx == 2) {
              outArr[yi][xs][outDof] = (inArr[yi][xs + 1][inDof] - L1[gDof])/4.0;
              outArr[yi][xs + 1][outDof] = ((3.0*inArr[yi][xs + 1][inDof]) -
                  (4.0*inArr[yi][xs][inDof]) + L1[gDof])/4.0;
            } else {
              outArr[yi][xs][outDof] = (inArr[yi][xs + 1][inDof] - L1[gDof])/4.0;
              outArr[yi][xs + nx - 1][outDof] = ((3.0*inArr[yi][xs + nx - 1][inDof]) -
                  (4.0*inArr[yi][xs + nx - 2][inDof]) + inArr[yi][xs + nx - 3][inDof])/4.0;
            }
          } else {
            if(nx == 1) {
              outArr[yi][xs][outDof] = (R1[gDof] - L1[gDof])/4.0;
            } else {
              outArr[yi][xs][outDof] = (inArr[yi][xs + 1][inDof] - L1[gDof])/4.0;
              outArr[yi][xs + nx - 1][outDof] = (R1[gDof] - inArr[yi][xs + nx - 2][inDof])/4.0;
            }
          }
        }//end d
      }//end yi
    }

    //dy
    {
      int len = nx*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int xi = xs; xi < (xs + nx); ++xi) {
        for(int d = 0; d < K; ++d) {
          int dof = ((K - 1)*(K + 1)) + d;
          first[((xi - xs)*K) + d] = inArr[ys][xi][dof];
          last[((xi - xs)*K) + d] = inArr[ys + ny - 1][xi][dof];
        }//end d
      }//end xi
      MPI_Request sReq1;
      MPI_Request rReq1;
      MPI_Request sReq2;
      MPI_Request rReq2;
      if(rj > 0) {
        MPI_Irecv(&(L1[0]), len, MPI_DOUBLE, prevY, 2, comm, &rReq2);
        MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevY, 1, comm, &sReq1);
      }
      if(rj < (py - 1)) {
        MPI_Irecv(&(R1[0]), len, MPI_DOUBLE, nextY, 1, comm, &rReq1);
        MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextY, 2, comm, &sReq2);
      }
      for(int yi = ys + 1; yi < (ys + ny - 1); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int d = 0; d < K; ++d) {
            int outDof = (K*(K + 1)) + d;
            int inDof = ((K - 1)*(K + 1)) + d;
            outArr[yi][xi][outDof] = (inArr[yi + 1][xi][inDof] - inArr[yi - 1][xi][inDof])/4.0;
          }//end d
        }//end xi
      }//end yi
      MPI_Status status;
      if(rj > 0) {
        MPI_Wait(&sReq1, &status);
        MPI_Wait(&rReq2, &status);
      }
      if(rj < (py - 1)) {
        MPI_Wait(&sReq2, &status);
        MPI_Wait(&rReq1, &status);
      }
      MPI_Request sReq3;
      MPI_Request rReq3;
      MPI_Request sReq4;
      MPI_Request rReq4;
      if((ys == 0) && (ny == 1)) {
        MPI_Irecv(&(R2[0]), len, MPI_DOUBLE, nextY, 3, comm, &rReq3);
      }
      if(((ys + ny) == Ny) && (ny == 1)) {
        MPI_Irecv(&(L2[0]), len, MPI_DOUBLE, prevY, 4, comm, &rReq4);
      }
      if(ys == 1) {
        if(ny == 1) {
          MPI_Isend(&(R1[0]), len, MPI_DOUBLE, prevY, 3, comm, &sReq3);
        } else {
          for(int xi = xs; xi < (xs + nx); ++xi) {
            for(int d = 0; d < K; ++d) {
              int dof = ((K - 1)*(K + 1)) + d;
              first[((xi - xs)*K) + d] = inArr[ys + 1][xi][dof];
            }//end d
          }//end yi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevY, 3, comm, &sReq3);
        }
      }
      if((ys + ny) == (Ny - 1)) {
        if(ny == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextY, 4, comm, &sReq4);
        } else {
          for(int xi = xs; xi < (xs + nx); ++xi) {
            for(int d = 0; d < K; ++d) {
              int dof = ((K - 1)*(K + 1)) + d;
              last[((xi - xs)*K) + d] = inArr[ys + ny - 2][xi][dof];
            }//end d
          }//end yi
          MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextY, 4, comm, &sReq4);
        }
      }
      if((ys == 0) && (ny == 1)) {
        MPI_Wait(&rReq3, &status);
      }
      if(((ys + ny) == Ny) && (ny == 1)) {
        MPI_Wait(&rReq4, &status);
      }
      if(ys == 1) {
        MPI_Wait(&sReq3, &status);
      }
      if((ys + ny) == (Ny - 1)) {
        MPI_Wait(&sReq4, &status);
      }
      for(int xi = xs; xi < (xs + nx); ++xi) {
        for(int d = 0; d < K; ++d) {
          int outDof = (K*(K + 1)) + d;
          int inDof = ((K - 1)*(K + 1)) + d;
          int gDof = ((xi - xs)*K) + d;
          if(rj == 0) {
            if(ny == 1) {
              outArr[ys][xi][outDof] = -((3.0*inArr[ys][xi][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
            } else if(ny == 2) {
              outArr[ys][xi][outDof] = -((3.0*inArr[ys][xi][inDof]) -
                  (4.0*inArr[ys + 1][xi][inDof]) + R1[gDof])/4.0;
              outArr[ys + 1][xi][outDof] = (R1[gDof] - inArr[ys][xi][inDof])/4.0;
            } else {
              outArr[ys][xi][outDof] = -((3.0*inArr[ys][xi][inDof]) - (4.0*inArr[ys + 1][xi][inDof])
                  + inArr[ys + 2][xi][inDof])/4.0;
              if(py == 1) {
                outArr[ys + ny - 1][xi][outDof] = ((3.0*inArr[ys + ny - 1][xi][inDof]) -
                    (4.0*inArr[ys + ny - 2][xi][inDof]) + inArr[ys + ny - 3][xi][inDof])/4.0;
              } else {
                outArr[ys + ny - 1][xi][outDof] = (R1[gDof] - inArr[ys + ny - 2][xi][inDof])/4.0;
              }
            }
          } else if(rj == (py - 1)) {
            if(ny == 1) {
              outArr[ys][xi][outDof] = ((3.0*inArr[ys][xi][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
            } else if(ny == 2) {
              outArr[ys][xi][outDof] = (inArr[ys + 1][xi][inDof] - L1[gDof])/4.0;
              outArr[ys + 1][xi][outDof] = ((3.0*inArr[ys + 1][xi][inDof]) -
                  (4.0*inArr[ys][xi][inDof]) + L1[gDof])/4.0;
            } else {
              outArr[ys][xi][outDof] = (inArr[ys + 1][xi][inDof] - L1[gDof])/4.0;
              outArr[ys + ny - 1][xi][outDof] = ((3.0*inArr[ys + ny - 1][xi][inDof]) -
                  (4.0*inArr[ys + ny - 2][xi][inDof]) + inArr[ys + ny - 3][xi][inDof])/4.0;
            }
          } else {
            if(ny == 1) {
              outArr[ys][xi][outDof] = (R1[gDof] - L1[gDof])/4.0;
            } else {
              outArr[ys][xi][outDof] = (inArr[ys + 1][xi][inDof] - L1[gDof])/4.0;
              outArr[ys + ny - 1][xi][outDof] = (R1[gDof] - inArr[ys + ny - 2][xi][inDof])/4.0;
            }
          }
        }//end d
      }//end xi
    }

    //dxdy
    {
      std::vector<double> L1(ny);
      std::vector<double> L2(ny);
      std::vector<double> R1(ny);
      std::vector<double> R2(ny);
      std::vector<double> first(ny);
      std::vector<double> last(ny);
      for(int yi = ys; yi < (ys + ny); ++yi) {
        int dof = (K*(K + 1)) + K - 1;
        first[yi - ys] = outArr[yi][xs][dof];
        last[yi - ys] = outArr[yi][xs + nx - 1][dof];
      }//end yi
      MPI_Request sReq1;
      MPI_Request rReq1;
      MPI_Request sReq2;
      MPI_Request rReq2;
      if(ri > 0) {
        MPI_Irecv(&(L1[0]), ny, MPI_DOUBLE, prevX, 2, comm, &rReq2);
        MPI_Isend(&(first[0]), ny, MPI_DOUBLE, prevX, 1, comm, &sReq1);
      }
      if(ri < (px - 1)) {
        MPI_Irecv(&(R1[0]), ny, MPI_DOUBLE, nextX, 1, comm, &rReq1);
        MPI_Isend(&(last[0]), ny, MPI_DOUBLE, nextX, 2, comm, &sReq2);
      }
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int xi = xs + 1; xi < (xs + nx - 1); ++xi) {
          int outDof = (K*(K + 1)) + K;
          int inDof = (K*(K + 1)) + K - 1;
          outArr[yi][xi][outDof] = (outArr[yi][xi + 1][inDof] - outArr[yi][xi - 1][inDof])/4.0;
        }//end xi
      }//end yi
      MPI_Status status;
      if(ri > 0) {
        MPI_Wait(&sReq1, &status);
        MPI_Wait(&rReq2, &status);
      }
      if(ri < (px - 1)) {
        MPI_Wait(&sReq2, &status);
        MPI_Wait(&rReq1, &status);
      }
      MPI_Request sReq3;
      MPI_Request rReq3;
      MPI_Request sReq4;
      MPI_Request rReq4;
      if((xs == 0) && (nx == 1)) {
        MPI_Irecv(&(R2[0]), ny, MPI_DOUBLE, nextX, 3, comm, &rReq3);
      }
      if(((xs + nx) == Nx) && (nx == 1)) {
        MPI_Irecv(&(L2[0]), ny, MPI_DOUBLE, prevX, 4, comm, &rReq4);
      }
      if(xs == 1) {
        if(nx == 1) {
          MPI_Isend(&(R1[0]), ny, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        } else {
          for(int yi = ys; yi < (ys + ny); ++yi) {
            int dof = (K*(K + 1)) + K - 1;
            first[yi - ys] = outArr[yi][xs + 1][dof];
          }//end yi
          MPI_Isend(&(first[0]), ny, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        }
      }
      if((xs + nx) == (Nx - 1)) {
        if(nx == 1) {
          MPI_Isend(&(L1[0]), ny, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        } else {
          for(int yi = ys; yi < (ys + ny); ++yi) {
            int dof = (K*(K + 1)) + K - 1;
            last[yi - ys] = outArr[yi][xs + nx - 2][dof];
          }//end yi
          MPI_Isend(&(last[0]), ny, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        }
      }
      if((xs == 0) && (nx == 1)) {
        MPI_Wait(&rReq3, &status);
      }
      if(((xs + nx) == Nx) && (nx == 1)) {
        MPI_Wait(&rReq4, &status);
      }
      if(xs == 1) {
        MPI_Wait(&sReq3, &status);
      }
      if((xs + nx) == (Nx - 1)) {
        MPI_Wait(&sReq4, &status);
      }
    }

    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  } else {
    int rk = rank/(px*py);
    int rj = (rank/px)%py;
    int ri = rank%px;
    int prevX = (((rk*py) + rj)*px) + (ri - 1);
    int nextX = (((rk*py) + rj)*px) + (ri + 1);
    int prevY = (((rk*py) + (rj - 1))*px) + ri;
    int nextY = (((rk*py) + (rj + 1))*px) + ri;
    int prevZ = ((((rk - 1)*py) + rj)*px) + ri;
    int nextZ = ((((rk + 1)*py) + rj)*px) + ri;
    PetscScalar**** inArr;
    PetscScalar**** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);

    //dx
    {
    }

    //dy
    {
    }

    //dz
    {
    }

    //dxdy
    {
    }

    //dydz
    {
    }

    //dzdx
    {
    }

    //dxdydz
    {
    }

    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  }
}


