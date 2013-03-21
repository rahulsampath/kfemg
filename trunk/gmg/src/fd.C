
#include "gmg/include/fd.h"
#include <vector>

#ifdef DEBUG
#include <cassert>
#endif

void applyFD(DM da, int K, Vec in, Vec out) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)da), &comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  PetscInt px;
  PetscInt py; 
  PetscInt pz; 
  DMDAGetInfo(da, &dim, &Nx, &Ny, &Nz, &px, &py, &pz,
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
      for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
        for(int d = 0; d < K; ++d, ++cnt) {
          int dof = (d*(K + 1)) + K - 1;
          first[cnt] = inArr[yi][xs][dof];
          last[cnt] = inArr[yi][xs + nx - 1][dof];
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
          for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
            for(int d = 0; d < K; ++d, ++cnt) {
              int dof = (d*(K + 1)) + K - 1;
              first[cnt] = inArr[yi][xs + 1][dof];
            }//end d
          }//end yi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        }
      }
      if((xs + nx) == (Nx - 1)) {
        if(nx == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        } else {
          for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
            for(int d = 0; d < K; ++d, ++cnt) {
              int dof = (d*(K + 1)) + K - 1;
              last[cnt] = inArr[yi][xs + nx - 2][dof];
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
      for(int yi = ys, gDof = 0; yi < (ys + ny); ++yi) {
        for(int d = 0; d < K; ++d, ++gDof) {
          int outDof = (d*(K + 1)) + K;
          int inDof = (d*(K + 1)) + K - 1;
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
      for(int xi = xs, cnt = 0; xi < (xs + nx); ++xi) {
        for(int d = 0; d < K; ++d, ++cnt) {
          int dof = ((K - 1)*(K + 1)) + d;
          first[cnt] = inArr[ys][xi][dof];
          last[cnt] = inArr[ys + ny - 1][xi][dof];
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
          for(int xi = xs, cnt = 0; xi < (xs + nx); ++xi) {
            for(int d = 0; d < K; ++d, ++cnt) {
              int dof = ((K - 1)*(K + 1)) + d;
              first[cnt] = inArr[ys + 1][xi][dof];
            }//end d
          }//end xi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevY, 3, comm, &sReq3);
        }
      }
      if((ys + ny) == (Ny - 1)) {
        if(ny == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextY, 4, comm, &sReq4);
        } else {
          for(int xi = xs, cnt = 0; xi < (xs + nx); ++xi) {
            for(int d = 0; d < K; ++d, ++cnt) {
              int dof = ((K - 1)*(K + 1)) + d;
              last[cnt] = inArr[ys + ny - 2][xi][dof];
            }//end d
          }//end xi
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
      for(int xi = xs, gDof = 0; xi < (xs + nx); ++xi) {
        for(int d = 0; d < K; ++d, ++gDof) {
          int outDof = (K*(K + 1)) + d;
          int inDof = ((K - 1)*(K + 1)) + d;
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
      for(int yi = ys; yi < (ys + ny); ++yi) {
        int outDof = (K*(K + 1)) + K;
        int inDof = (K*(K + 1)) + K - 1;
        int gDof = yi - ys;
        if(ri == 0) {
          if(nx == 1) {
            outArr[yi][xs][outDof] = -((3.0*outArr[yi][xs][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
          } else if(nx == 2) {
            outArr[yi][xs][outDof] = -((3.0*outArr[yi][xs][inDof]) -
                (4.0*outArr[yi][xs + 1][inDof]) + R1[gDof])/4.0;
            outArr[yi][xs + 1][outDof] = (R1[gDof] - outArr[yi][xs][inDof])/4.0;
          } else {
            outArr[yi][xs][outDof] = -((3.0*outArr[yi][xs][inDof]) - (4.0*outArr[yi][xs + 1][inDof])
                + outArr[yi][xs + 2][inDof])/4.0;
            if(px == 1) {
              outArr[yi][xs + nx - 1][outDof] = ((3.0*outArr[yi][xs + nx - 1][inDof]) -
                  (4.0*outArr[yi][xs + nx - 2][inDof]) + outArr[yi][xs + nx - 3][inDof])/4.0;
            } else {
              outArr[yi][xs + nx - 1][outDof] = (R1[gDof] - outArr[yi][xs + nx - 2][inDof])/4.0;
            }
          }
        } else if(ri == (px - 1)) {
          if(nx == 1) {
            outArr[yi][xs][outDof] = ((3.0*outArr[yi][xs][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
          } else if(nx == 2) {
            outArr[yi][xs][outDof] = (outArr[yi][xs + 1][inDof] - L1[gDof])/4.0;
            outArr[yi][xs + 1][outDof] = ((3.0*outArr[yi][xs + 1][inDof]) -
                (4.0*outArr[yi][xs][inDof]) + L1[gDof])/4.0;
          } else {
            outArr[yi][xs][outDof] = (outArr[yi][xs + 1][inDof] - L1[gDof])/4.0;
            outArr[yi][xs + nx - 1][outDof] = ((3.0*outArr[yi][xs + nx - 1][inDof]) -
                (4.0*outArr[yi][xs + nx - 2][inDof]) + outArr[yi][xs + nx - 3][inDof])/4.0;
          }
        } else {
          if(nx == 1) {
            outArr[yi][xs][outDof] = (R1[gDof] - L1[gDof])/4.0;
          } else {
            outArr[yi][xs][outDof] = (outArr[yi][xs + 1][inDof] - L1[gDof])/4.0;
            outArr[yi][xs + nx - 1][outDof] = (R1[gDof] - outArr[yi][xs + nx - 2][inDof])/4.0;
          }
        }
      }//end yi
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
      int len = ny*nz*K*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int dz = 0; dz < K; ++dz) {
            for(int dy = 0; dy < K; ++dy, ++cnt) {
              int dof = (((dz*(K + 1)) + dy)*(K + 1)) + K - 1;
              first[cnt] = inArr[zi][yi][xs][dof];
              last[cnt] = inArr[zi][yi][xs + nx - 1][dof];
            }//end dy
          }//end dz
        }//end yi
      }//end zi
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
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int xi = xs + 1; xi < (xs + nx - 1); ++xi) {
            for(int dz = 0; dz < K; ++dz) {
              for(int dy = 0; dy < K; ++dy) {
                int outDof = (((dz*(K + 1)) + dy)*(K + 1)) + K;
                int inDof = (((dz*(K + 1)) + dy)*(K + 1)) + K - 1;
                outArr[zi][yi][xi][outDof] = (inArr[zi][yi][xi + 1][inDof] - inArr[zi][yi][xi - 1][inDof])/4.0;
              }//end dy
            }//end dz
          }//end xi
        }//end yi
      }//end zi
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
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int yi = ys; yi < (ys + ny); ++yi) {
              for(int dz = 0; dz < K; ++dz) {
                for(int dy = 0; dy < K; ++dy, ++cnt) {
                  int dof = (((dz*(K + 1)) + dy)*(K + 1)) + K - 1;
                  first[cnt] = inArr[zi][yi][xs + 1][dof];
                }//end dy
              }//end dz
            }//end yi
          }//end zi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        }
      }
      if((xs + nx) == (Nx - 1)) {
        if(nx == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        } else {
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int yi = ys; yi < (ys + ny); ++yi) {
              for(int dz = 0; dz < K; ++dz) {
                for(int dy = 0; dy < K; ++dy, ++cnt) {
                  int dof = (((dz*(K + 1)) + dy)*(K + 1)) + K - 1;
                  last[cnt] = inArr[zi][yi][xs + nx - 2][dof];
                }//end dy
              }//end dz
            }//end yi
          }//end zi
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
      for(int zi = zs, gDof = 0; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int dz = 0; dz < K; ++dz) {
            for(int dy = 0; dy < K; ++dy, ++gDof) {
              int outDof = (((dz*(K + 1)) + dy)*(K + 1)) + K;
              int inDof = (((dz*(K + 1)) + dy)*(K + 1)) + K - 1;
              if(ri == 0) {
                if(nx == 1) {
                  outArr[zi][yi][xs][outDof] = -((3.0*inArr[zi][yi][xs][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
                } else if(nx == 2) {
                  outArr[zi][yi][xs][outDof] = -((3.0*inArr[zi][yi][xs][inDof]) -
                      (4.0*inArr[zi][yi][xs + 1][inDof]) + R1[gDof])/4.0;
                  outArr[zi][yi][xs + 1][outDof] = (R1[gDof] - inArr[zi][yi][xs][inDof])/4.0;
                } else {
                  outArr[zi][yi][xs][outDof] = -((3.0*inArr[zi][yi][xs][inDof]) - (4.0*inArr[zi][yi][xs + 1][inDof])
                      + inArr[zi][yi][xs + 2][inDof])/4.0;
                  if(px == 1) {
                    outArr[zi][yi][xs + nx - 1][outDof] = ((3.0*inArr[zi][yi][xs + nx - 1][inDof]) -
                        (4.0*inArr[zi][yi][xs + nx - 2][inDof]) + inArr[zi][yi][xs + nx - 3][inDof])/4.0;
                  } else {
                    outArr[zi][yi][xs + nx - 1][outDof] = (R1[gDof] - inArr[zi][yi][xs + nx - 2][inDof])/4.0;
                  }
                }
              } else if(ri == (px - 1)) {
                if(nx == 1) {
                  outArr[zi][yi][xs][outDof] = ((3.0*inArr[zi][yi][xs][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
                } else if(nx == 2) {
                  outArr[zi][yi][xs][outDof] = (inArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
                  outArr[zi][yi][xs + 1][outDof] = ((3.0*inArr[zi][yi][xs + 1][inDof]) -
                      (4.0*inArr[zi][yi][xs][inDof]) + L1[gDof])/4.0;
                } else {
                  outArr[zi][yi][xs][outDof] = (inArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
                  outArr[zi][yi][xs + nx - 1][outDof] = ((3.0*inArr[zi][yi][xs + nx - 1][inDof]) -
                      (4.0*inArr[zi][yi][xs + nx - 2][inDof]) + inArr[zi][yi][xs + nx - 3][inDof])/4.0;
                }
              } else {
                if(nx == 1) {
                  outArr[zi][yi][xs][outDof] = (R1[gDof] - L1[gDof])/4.0;
                } else {
                  outArr[zi][yi][xs][outDof] = (inArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
                  outArr[zi][yi][xs + nx - 1][outDof] = (R1[gDof] - inArr[zi][yi][xs + nx - 2][inDof])/4.0;
                }
              }
            }//end dy
          }//end dz
        }//end yi
      }//end zi
    }

    //dy
    {
      int len = nx*nz*K*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dz = 0; dz < K; ++dz) {
            for(int dx = 0; dx < K; ++dx, ++cnt) {
              int dof = (((dz*(K + 1)) + (K - 1))*(K + 1)) + dx;
              first[cnt] = inArr[zi][ys][xi][dof];
              last[cnt] = inArr[zi][ys + ny - 1][xi][dof];
            }//end dx
          }//end dz
        }//end xi
      }//end zi
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
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int yi = ys + 1; yi < (ys + ny - 1); ++yi) {
          for(int xi = xs; xi < (xs + nx); ++xi) {
            for(int dz = 0; dz < K; ++dz) {
              for(int dx = 0; dx < K; ++dx) {
                int outDof = (((dz*(K + 1)) + K)*(K + 1)) + dx;
                int inDof = (((dz*(K + 1)) + (K - 1))*(K + 1)) + dx;
                outArr[zi][yi][xi][outDof] = (inArr[zi][yi + 1][xi][inDof] - inArr[zi][yi - 1][xi][inDof])/4.0;
              }//end dx
            }//end dz
          }//end xi
        }//end yi
      }//end zi
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
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dz = 0; dz < K; ++dz) {
                for(int dx = 0; dx < K; ++dx, ++cnt) {
                  int dof = (((dz*(K + 1)) + (K - 1))*(K + 1)) + dx;
                  first[cnt] = inArr[zi][ys + 1][xi][dof];
                }//end dx
              }//end dz
            }//end xi
          }//end zi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevY, 3, comm, &sReq3);
        }
      }
      if((ys + ny) == (Ny - 1)) {
        if(ny == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextY, 4, comm, &sReq4);
        } else {
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dz = 0; dz < K; ++dz) {
                for(int dx = 0; dx < K; ++dx, ++cnt) {
                  int dof = (((dz*(K + 1)) + (K - 1))*(K + 1)) + dx;
                  last[cnt] = inArr[zi][ys + ny - 2][xi][dof];
                }//end dx
              }//end dz
            }//end xi
          }//end zi
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
      for(int zi = zs, gDof = 0; zi < (zs + nz); ++zi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dz = 0; dz < K; ++dz) {
            for(int dx = 0; dx < K; ++dx, ++gDof) {
              int outDof = (((dz*(K + 1)) + K)*(K + 1)) + dx;
              int inDof = (((dz*(K + 1)) + (K - 1))*(K + 1)) + dx;
              if(rj == 0) {
                if(ny == 1) {
                  outArr[zi][ys][xi][outDof] = -((3.0*inArr[zi][ys][xi][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
                } else if(ny == 2) {
                  outArr[zi][ys][xi][outDof] = -((3.0*inArr[zi][ys][xi][inDof]) -
                      (4.0*inArr[zi][ys + 1][xi][inDof]) + R1[gDof])/4.0;
                  outArr[zi][ys + 1][xi][outDof] = (R1[gDof] - inArr[zi][ys][xi][inDof])/4.0;
                } else {
                  outArr[zi][ys][xi][outDof] = -((3.0*inArr[zi][ys][xi][inDof]) - (4.0*inArr[zi][ys + 1][xi][inDof])
                      + inArr[zi][ys + 2][xi][inDof])/4.0;
                  if(py == 1) {
                    outArr[zi][ys + ny - 1][xi][outDof] = ((3.0*inArr[zi][ys + ny - 1][xi][inDof]) -
                        (4.0*inArr[zi][ys + ny - 2][xi][inDof]) + inArr[zi][ys + ny - 3][xi][inDof])/4.0;
                  } else {
                    outArr[zi][ys + ny - 1][xi][outDof] = (R1[gDof] - inArr[zi][ys + ny - 2][xi][inDof])/4.0;
                  }
                }
              } else if(rj == (py - 1)) {
                if(ny == 1) {
                  outArr[zi][ys][xi][outDof] = ((3.0*inArr[zi][ys][xi][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
                } else if(ny == 2) {
                  outArr[zi][ys][xi][outDof] = (inArr[zi][ys + 1][xi][inDof] - L1[gDof])/4.0;
                  outArr[zi][ys + 1][xi][outDof] = ((3.0*inArr[zi][ys + 1][xi][inDof]) -
                      (4.0*inArr[zi][ys][xi][inDof]) + L1[gDof])/4.0;
                } else {
                  outArr[zi][ys][xi][outDof] = (inArr[zi][ys + 1][xi][inDof] - L1[gDof])/4.0;
                  outArr[zi][ys + ny - 1][xi][outDof] = ((3.0*inArr[zi][ys + ny - 1][xi][inDof]) -
                      (4.0*inArr[zi][ys + ny - 2][xi][inDof]) + inArr[zi][ys + ny - 3][xi][inDof])/4.0;
                }
              } else {
                if(ny == 1) {
                  outArr[zi][ys][xi][outDof] = (R1[gDof] - L1[gDof])/4.0;
                } else {
                  outArr[zi][ys][xi][outDof] = (inArr[zi][ys + 1][xi][inDof] - L1[gDof])/4.0;
                  outArr[zi][ys + ny - 1][xi][outDof] = (R1[gDof] - inArr[zi][ys + ny - 2][xi][inDof])/4.0;
                }
              }
            }//end dx
          }//end dz
        }//end xi
      }//end zi
    }

    //dz
    {
      int len = nx*ny*K*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dy = 0; dy < K; ++dy) {
            for(int dx = 0; dx < K; ++dx, ++cnt) {
              int dof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + dx;
              first[cnt] = inArr[zs][yi][xi][dof];
              last[cnt] = inArr[zs + nz - 1][yi][xi][dof];
            }//end dx
          }//end dy
        }//end xi
      }//end yi
      MPI_Request sReq1;
      MPI_Request rReq1;
      MPI_Request sReq2;
      MPI_Request rReq2;
      if(rk > 0) {
        MPI_Irecv(&(L1[0]), len, MPI_DOUBLE, prevZ, 2, comm, &rReq2);
        MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevZ, 1, comm, &sReq1);
      }
      if(rk < (pz - 1)) {
        MPI_Irecv(&(R1[0]), len, MPI_DOUBLE, nextZ, 1, comm, &rReq1);
        MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextZ, 2, comm, &sReq2);
      }
      for(int zi = zs + 1; zi < (zs + nz - 1); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int xi = xs; xi < (xs + nx); ++xi) {
            for(int dy = 0; dy < K; ++dy) {
              for(int dx = 0; dx < K; ++dx) {
                int outDof = (((K*(K + 1)) + dy)*(K + 1)) + dx;
                int inDof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + dx;
                outArr[zi][yi][xi][outDof] = (inArr[zi + 1][yi][xi][inDof] - inArr[zi - 1][yi][xi][inDof])/4.0;
              }//end dx
            }//end dy
          }//end xi
        }//end yi
      }//end zi
      MPI_Status status;
      if(rk > 0) {
        MPI_Wait(&sReq1, &status);
        MPI_Wait(&rReq2, &status);
      }
      if(rk < (pz - 1)) {
        MPI_Wait(&sReq2, &status);
        MPI_Wait(&rReq1, &status);
      }
      MPI_Request sReq3;
      MPI_Request rReq3;
      MPI_Request sReq4;
      MPI_Request rReq4;
      if((zs == 0) && (nz == 1)) {
        MPI_Irecv(&(R2[0]), len, MPI_DOUBLE, nextZ, 3, comm, &rReq3);
      }
      if(((zs + nz) == Nz) && (nz == 1)) {
        MPI_Irecv(&(L2[0]), len, MPI_DOUBLE, prevZ, 4, comm, &rReq4);
      }
      if(zs == 1) {
        if(nz == 1) {
          MPI_Isend(&(R1[0]), len, MPI_DOUBLE, prevZ, 3, comm, &sReq3);
        } else {
          for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dy = 0; dy < K; ++dy) {
                for(int dx = 0; dx < K; ++dx, ++cnt) {
                  int dof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + dx;
                  first[cnt] = inArr[zs + 1][yi][xi][dof];
                }//end dx
              }//end dy
            }//end xi
          }//end yi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevZ, 3, comm, &sReq3);
        }
      }
      if((zs + nz) == (Nz - 1)) {
        if(nz == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextZ, 4, comm, &sReq4);
        } else {
          for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dy = 0; dy < K; ++dy) {
                for(int dx = 0; dx < K; ++dx, ++cnt) {
                  int dof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + dx;
                  last[cnt] = inArr[zs + nz - 2][yi][xi][dof];
                }//end dx
              }//end dy
            }//end xi
          }//end yi
          MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextZ, 4, comm, &sReq4);
        }
      }
      if((zs == 0) && (nz == 1)) {
        MPI_Wait(&rReq3, &status);
      }
      if(((zs + nz) == Nz) && (nz == 1)) {
        MPI_Wait(&rReq4, &status);
      }
      if(zs == 1) {
        MPI_Wait(&sReq3, &status);
      }
      if((zs + nz) == (Nz - 1)) {
        MPI_Wait(&sReq4, &status);
      }
      for(int yi = ys, gDof = 0; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dy = 0; dy < K; ++dy) {
            for(int dx = 0; dx < K; ++dx, ++gDof) {
              int outDof = (((K*(K + 1)) + dy)*(K + 1)) + dx;
              int inDof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + dx;
              if(rk == 0) {
                if(nz == 1) {
                  outArr[zs][yi][xi][outDof] = -((3.0*inArr[zs][yi][xi][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
                } else if(nz == 2) {
                  outArr[zs][yi][xi][outDof] = -((3.0*inArr[zs][yi][xi][inDof]) -
                      (4.0*inArr[zs + 1][yi][xi][inDof]) + R1[gDof])/4.0;
                  outArr[zs + 1][yi][xi][outDof] = (R1[gDof] - inArr[zs][yi][xi][inDof])/4.0;
                } else {
                  outArr[zs][yi][xi][outDof] = -((3.0*inArr[zs][yi][xi][inDof]) - (4.0*inArr[zs + 1][yi][xi][inDof])
                      + inArr[zs + 2][yi][xi][inDof])/4.0;
                  if(pz == 1) {
                    outArr[zs + nz - 1][yi][xi][outDof] = ((3.0*inArr[zs + nz - 1][yi][xi][inDof]) -
                        (4.0*inArr[zs + nz - 2][yi][xi][inDof]) + inArr[zs + nz - 3][yi][xi][inDof])/4.0;
                  } else {
                    outArr[zs + nz - 1][yi][xi][outDof] = (R1[gDof] - inArr[zs + nz - 2][yi][xi][inDof])/4.0;
                  }
                }
              } else if(rk == (pz - 1)) {
                if(nz == 1) {
                  outArr[zs][yi][xi][outDof] = ((3.0*inArr[zs][yi][xi][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
                } else if(nz == 2) {
                  outArr[zs][yi][xi][outDof] = (inArr[zs + 1][yi][xi][inDof] - L1[gDof])/4.0;
                  outArr[zs + 1][yi][xi][outDof] = ((3.0*inArr[zs + 1][yi][xi][inDof]) -
                      (4.0*inArr[zs][yi][xi][inDof]) + L1[gDof])/4.0;
                } else {
                  outArr[zs][yi][xi][outDof] = (inArr[zs + 1][yi][xi][inDof] - L1[gDof])/4.0;
                  outArr[zs + nz - 1][yi][xi][outDof] = ((3.0*inArr[zs + nz - 1][yi][xi][inDof]) -
                      (4.0*inArr[zs + nz - 2][yi][xi][inDof]) + inArr[zs + nz - 3][yi][xi][inDof])/4.0;
                }
              } else {
                if(nz == 1) {
                  outArr[zs][yi][xi][outDof] = (R1[gDof] - L1[gDof])/4.0;
                } else {
                  outArr[zs][yi][xi][outDof] = (inArr[zs + 1][yi][xi][inDof] - L1[gDof])/4.0;
                  outArr[zs + nz - 1][yi][xi][outDof] = (R1[gDof] - inArr[zs + nz - 2][yi][xi][inDof])/4.0;
                }
              }
            }//end dx
          }//end dy
        }//end xi
      }//end yi
    }

    //dxdy
    {
      int len = ny*nz*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int dz = 0; dz < K; ++dz, ++cnt) {
            int dof = (((dz*(K + 1)) + K)*(K + 1)) + K - 1;
            first[cnt] = outArr[zi][yi][xs][dof];
            last[cnt] = outArr[zi][yi][xs + nx - 1][dof];
          }//end dz
        }//end yi
      }//end zi
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
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int xi = xs + 1; xi < (xs + nx - 1); ++xi) {
            for(int dz = 0; dz < K; ++dz) {
              int outDof = (((dz*(K + 1)) + K)*(K + 1)) + K;
              int inDof = (((dz*(K + 1)) + K)*(K + 1)) + K - 1;
              outArr[zi][yi][xi][outDof] = (outArr[zi][yi][xi + 1][inDof] - outArr[zi][yi][xi - 1][inDof])/4.0;
            }//end dz
          }//end xi
        }//end yi
      }//end zi
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
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int yi = ys; yi < (ys + ny); ++yi) {
              for(int dz = 0; dz < K; ++dz, ++cnt) {
                int dof = (((dz*(K + 1)) + K)*(K + 1)) + K - 1;
                first[cnt] = outArr[zi][yi][xs + 1][dof];
              }//end dz
            }//end yi
          }//end zi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        }
      }
      if((xs + nx) == (Nx - 1)) {
        if(nx == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        } else {
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int yi = ys; yi < (ys + ny); ++yi) {
              for(int dz = 0; dz < K; ++dz, ++cnt) {
                int dof = (((dz*(K + 1)) + K)*(K + 1)) + K - 1;
                last[cnt] = outArr[zi][yi][xs + nx - 2][dof];
              }//end dz
            }//end yi
          }//end zi
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
      for(int zi = zs, gDof = 0; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int dz = 0; dz < K; ++dz, ++gDof) {
            int outDof = (((dz*(K + 1)) + K)*(K + 1)) + K;
            int inDof = (((dz*(K + 1)) + K)*(K + 1)) + K - 1;
            if(ri == 0) {
              if(nx == 1) {
                outArr[zi][yi][xs][outDof] = -((3.0*outArr[zi][yi][xs][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
              } else if(nx == 2) {
                outArr[zi][yi][xs][outDof] = -((3.0*outArr[zi][yi][xs][inDof]) -
                    (4.0*outArr[zi][yi][xs + 1][inDof]) + R1[gDof])/4.0;
                outArr[zi][yi][xs + 1][outDof] = (R1[gDof] - outArr[zi][yi][xs][inDof])/4.0;
              } else {
                outArr[zi][yi][xs][outDof] = -((3.0*outArr[zi][yi][xs][inDof]) - (4.0*outArr[zi][yi][xs + 1][inDof])
                    + outArr[zi][yi][xs + 2][inDof])/4.0;
                if(px == 1) {
                  outArr[zi][yi][xs + nx - 1][outDof] = ((3.0*outArr[zi][yi][xs + nx - 1][inDof]) -
                      (4.0*outArr[zi][yi][xs + nx - 2][inDof]) + outArr[zi][yi][xs + nx - 3][inDof])/4.0;
                } else {
                  outArr[zi][yi][xs + nx - 1][outDof] = (R1[gDof] - outArr[zi][yi][xs + nx - 2][inDof])/4.0;
                }
              }
            } else if(ri == (px - 1)) {
              if(nx == 1) {
                outArr[zi][yi][xs][outDof] = ((3.0*outArr[zi][yi][xs][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
              } else if(nx == 2) {
                outArr[zi][yi][xs][outDof] = (outArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
                outArr[zi][yi][xs + 1][outDof] = ((3.0*outArr[zi][yi][xs + 1][inDof]) -
                    (4.0*outArr[zi][yi][xs][inDof]) + L1[gDof])/4.0;
              } else {
                outArr[zi][yi][xs][outDof] = (outArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
                outArr[zi][yi][xs + nx - 1][outDof] = ((3.0*outArr[zi][yi][xs + nx - 1][inDof]) -
                    (4.0*outArr[zi][yi][xs + nx - 2][inDof]) + outArr[zi][yi][xs + nx - 3][inDof])/4.0;
              }
            } else {
              if(nx == 1) {
                outArr[zi][yi][xs][outDof] = (R1[gDof] - L1[gDof])/4.0;
              } else {
                outArr[zi][yi][xs][outDof] = (outArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
                outArr[zi][yi][xs + nx - 1][outDof] = (R1[gDof] - outArr[zi][yi][xs + nx - 2][inDof])/4.0;
              }
            }
          }//end dz
        }//end yi
      }//end zi
    }

    //dydz
    {
      int len = nx*nz*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dx = 0; dx < K; ++dx, ++cnt) {
            int dof = (((K*(K + 1)) + (K - 1))*(K + 1)) + dx;
            first[cnt] = outArr[zi][ys][xi][dof];
            last[cnt] = outArr[zi][ys + ny - 1][xi][dof];
          }//end dx
        }//end xi
      }//end zi
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
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int yi = ys + 1; yi < (ys + ny - 1); ++yi) {
          for(int xi = xs; xi < (xs + nx); ++xi) {
            for(int dx = 0; dx < K; ++dx) {
              int outDof = (((K*(K + 1)) + K)*(K + 1)) + dx;
              int inDof = (((K*(K + 1)) + (K - 1))*(K + 1)) + dx;
              outArr[zi][yi][xi][outDof] = (outArr[zi][yi + 1][xi][inDof] - outArr[zi][yi - 1][xi][inDof])/4.0;
            }//end dx
          }//end xi
        }//end yi
      }//end zi
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
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dx = 0; dx < K; ++dx, ++cnt) {
                int dof = (((K*(K + 1)) + (K - 1))*(K + 1)) + dx;
                first[cnt] = outArr[zi][ys + 1][xi][dof];
              }//end dx
            }//end xi
          }//end zi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevY, 3, comm, &sReq3);
        }
      }
      if((ys + ny) == (Ny - 1)) {
        if(ny == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextY, 4, comm, &sReq4);
        } else {
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dx = 0; dx < K; ++dx, ++cnt) {
                int dof = (((K*(K + 1)) + (K - 1))*(K + 1)) + dx;
                last[cnt] = outArr[zi][ys + ny - 2][xi][dof];
              }//end dx
            }//end xi
          }//end zi
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
      for(int zi = zs, gDof = 0; zi < (zs + nz); ++zi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dx = 0; dx < K; ++dx, ++gDof) {
            int outDof = (((K*(K + 1)) + K)*(K + 1)) + dx;
            int inDof = (((K*(K + 1)) + (K - 1))*(K + 1)) + dx;
            if(rj == 0) {
              if(ny == 1) {
                outArr[zi][ys][xi][outDof] = -((3.0*outArr[zi][ys][xi][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
              } else if(ny == 2) {
                outArr[zi][ys][xi][outDof] = -((3.0*outArr[zi][ys][xi][inDof]) -
                    (4.0*outArr[zi][ys + 1][xi][inDof]) + R1[gDof])/4.0;
                outArr[zi][ys + 1][xi][outDof] = (R1[gDof] - outArr[zi][ys][xi][inDof])/4.0;
              } else {
                outArr[zi][ys][xi][outDof] = -((3.0*outArr[zi][ys][xi][inDof]) - (4.0*outArr[zi][ys + 1][xi][inDof])
                    + outArr[zi][ys + 2][xi][inDof])/4.0;
                if(py == 1) {
                  outArr[zi][ys + ny - 1][xi][outDof] = ((3.0*outArr[zi][ys + ny - 1][xi][inDof]) -
                      (4.0*outArr[zi][ys + ny - 2][xi][inDof]) + outArr[zi][ys + ny - 3][xi][inDof])/4.0;
                } else {
                  outArr[zi][ys + ny - 1][xi][outDof] = (R1[gDof] - outArr[zi][ys + ny - 2][xi][inDof])/4.0;
                }
              }
            } else if(rj == (py - 1)) {
              if(ny == 1) {
                outArr[zi][ys][xi][outDof] = ((3.0*outArr[zi][ys][xi][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
              } else if(ny == 2) {
                outArr[zi][ys][xi][outDof] = (outArr[zi][ys + 1][xi][inDof] - L1[gDof])/4.0;
                outArr[zi][ys + 1][xi][outDof] = ((3.0*outArr[zi][ys + 1][xi][inDof]) -
                    (4.0*outArr[zi][ys][xi][inDof]) + L1[gDof])/4.0;
              } else {
                outArr[zi][ys][xi][outDof] = (outArr[zi][ys + 1][xi][inDof] - L1[gDof])/4.0;
                outArr[zi][ys + ny - 1][xi][outDof] = ((3.0*outArr[zi][ys + ny - 1][xi][inDof]) -
                    (4.0*outArr[zi][ys + ny - 2][xi][inDof]) + outArr[zi][ys + ny - 3][xi][inDof])/4.0;
              }
            } else {
              if(ny == 1) {
                outArr[zi][ys][xi][outDof] = (R1[gDof] - L1[gDof])/4.0;
              } else {
                outArr[zi][ys][xi][outDof] = (outArr[zi][ys + 1][xi][inDof] - L1[gDof])/4.0;
                outArr[zi][ys + ny - 1][xi][outDof] = (R1[gDof] - outArr[zi][ys + ny - 2][xi][inDof])/4.0;
              }
            }
          }//end dx
        }//end xi
      }//end zi
    }

    //dzdx
    {
      int len = nx*ny*K;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dy = 0; dy < K; ++dy, ++cnt) {
            int dof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + K;
            first[cnt] = outArr[zs][yi][xi][dof];
            last[cnt] = outArr[zs + nz - 1][yi][xi][dof];
          }//end dy
        }//end xi
      }//end yi
      MPI_Request sReq1;
      MPI_Request rReq1;
      MPI_Request sReq2;
      MPI_Request rReq2;
      if(rk > 0) {
        MPI_Irecv(&(L1[0]), len, MPI_DOUBLE, prevZ, 2, comm, &rReq2);
        MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevZ, 1, comm, &sReq1);
      }
      if(rk < (pz - 1)) {
        MPI_Irecv(&(R1[0]), len, MPI_DOUBLE, nextZ, 1, comm, &rReq1);
        MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextZ, 2, comm, &sReq2);
      }
      for(int zi = zs + 1; zi < (zs + nz - 1); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int xi = xs; xi < (xs + nx); ++xi) {
            for(int dy = 0; dy < K; ++dy) {
              int outDof = (((K*(K + 1)) + dy)*(K + 1)) + K;
              int inDof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + K;
              outArr[zi][yi][xi][outDof] = (outArr[zi + 1][yi][xi][inDof] - outArr[zi - 1][yi][xi][inDof])/4.0;
            }//end dy
          }//end xi
        }//end yi
      }//end zi
      MPI_Status status;
      if(rk > 0) {
        MPI_Wait(&sReq1, &status);
        MPI_Wait(&rReq2, &status);
      }
      if(rk < (pz - 1)) {
        MPI_Wait(&sReq2, &status);
        MPI_Wait(&rReq1, &status);
      }
      MPI_Request sReq3;
      MPI_Request rReq3;
      MPI_Request sReq4;
      MPI_Request rReq4;
      if((zs == 0) && (nz == 1)) {
        MPI_Irecv(&(R2[0]), len, MPI_DOUBLE, nextZ, 3, comm, &rReq3);
      }
      if(((zs + nz) == Nz) && (nz == 1)) {
        MPI_Irecv(&(L2[0]), len, MPI_DOUBLE, prevZ, 4, comm, &rReq4);
      }
      if(zs == 1) {
        if(nz == 1) {
          MPI_Isend(&(R1[0]), len, MPI_DOUBLE, prevZ, 3, comm, &sReq3);
        } else {
          for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dy = 0; dy < K; ++dy, ++cnt) {
                int dof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + K;
                first[cnt] = outArr[zs + 1][yi][xi][dof];
              }//end dy
            }//end xi
          }//end yi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevZ, 3, comm, &sReq3);
        }
      }
      if((zs + nz) == (Nz - 1)) {
        if(nz == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextZ, 4, comm, &sReq4);
        } else {
          for(int yi = ys, cnt = 0; yi < (ys + ny); ++yi) {
            for(int xi = xs; xi < (xs + nx); ++xi) {
              for(int dy = 0; dy < K; ++dy, ++cnt) {
                int dof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + K;
                last[cnt] = outArr[zs + nz - 2][yi][xi][dof];
              }//end dy
            }//end xi
          }//end yi
          MPI_Isend(&(last[0]), len, MPI_DOUBLE, nextZ, 4, comm, &sReq4);
        }
      }
      if((zs == 0) && (nz == 1)) {
        MPI_Wait(&rReq3, &status);
      }
      if(((zs + nz) == Nz) && (nz == 1)) {
        MPI_Wait(&rReq4, &status);
      }
      if(zs == 1) {
        MPI_Wait(&sReq3, &status);
      }
      if((zs + nz) == (Nz - 1)) {
        MPI_Wait(&sReq4, &status);
      }
      for(int yi = ys, gDof = 0; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          for(int dy = 0; dy < K; ++dy, ++gDof) {
            int outDof = (((K*(K + 1)) + dy)*(K + 1)) + K;
            int inDof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + K;
            if(rk == 0) {
              if(nz == 1) {
                outArr[zs][yi][xi][outDof] = -((3.0*outArr[zs][yi][xi][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
              } else if(nz == 2) {
                outArr[zs][yi][xi][outDof] = -((3.0*outArr[zs][yi][xi][inDof]) -
                    (4.0*outArr[zs + 1][yi][xi][inDof]) + R1[gDof])/4.0;
                outArr[zs + 1][yi][xi][outDof] = (R1[gDof] - outArr[zs][yi][xi][inDof])/4.0;
              } else {
                outArr[zs][yi][xi][outDof] = -((3.0*outArr[zs][yi][xi][inDof]) - (4.0*outArr[zs + 1][yi][xi][inDof])
                    + outArr[zs + 2][yi][xi][inDof])/4.0;
                if(pz == 1) {
                  outArr[zs + nz - 1][yi][xi][outDof] = ((3.0*outArr[zs + nz - 1][yi][xi][inDof]) -
                      (4.0*outArr[zs + nz - 2][yi][xi][inDof]) + outArr[zs + nz - 3][yi][xi][inDof])/4.0;
                } else {
                  outArr[zs + nz - 1][yi][xi][outDof] = (R1[gDof] - outArr[zs + nz - 2][yi][xi][inDof])/4.0;
                }
              }
            } else if(rk == (pz - 1)) {
              if(nz == 1) {
                outArr[zs][yi][xi][outDof] = ((3.0*outArr[zs][yi][xi][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
              } else if(nz == 2) {
                outArr[zs][yi][xi][outDof] = (outArr[zs + 1][yi][xi][inDof] - L1[gDof])/4.0;
                outArr[zs + 1][yi][xi][outDof] = ((3.0*outArr[zs + 1][yi][xi][inDof]) -
                    (4.0*outArr[zs][yi][xi][inDof]) + L1[gDof])/4.0;
              } else {
                outArr[zs][yi][xi][outDof] = (outArr[zs + 1][yi][xi][inDof] - L1[gDof])/4.0;
                outArr[zs + nz - 1][yi][xi][outDof] = ((3.0*outArr[zs + nz - 1][yi][xi][inDof]) -
                    (4.0*outArr[zs + nz - 2][yi][xi][inDof]) + outArr[zs + nz - 3][yi][xi][inDof])/4.0;
              }
            } else {
              if(nz == 1) {
                outArr[zs][yi][xi][outDof] = (R1[gDof] - L1[gDof])/4.0;
              } else {
                outArr[zs][yi][xi][outDof] = (outArr[zs + 1][yi][xi][inDof] - L1[gDof])/4.0;
                outArr[zs + nz - 1][yi][xi][outDof] = (R1[gDof] - outArr[zs + nz - 2][yi][xi][inDof])/4.0;
              }
            }
          }//end dy
        }//end xi
      }//end yi
    }

    //dxdydz
    {
      int len = ny*nz;
      std::vector<double> L1(len);
      std::vector<double> L2(len);
      std::vector<double> R1(len);
      std::vector<double> R2(len);
      std::vector<double> first(len);
      std::vector<double> last(len);
      for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi, ++cnt) {
          int dof = (((K*(K + 1)) + K)*(K + 1)) + K - 1;
          first[cnt] = outArr[zi][yi][xs][dof];
          last[cnt] = outArr[zi][yi][xs + nx - 1][dof];
        }//end yi
      }//end zi
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
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          for(int xi = xs + 1; xi < (xs + nx - 1); ++xi) {
            int outDof = (((K*(K + 1)) + K)*(K + 1)) + K;
            int inDof = (((K*(K + 1)) + K)*(K + 1)) + K - 1;
            outArr[zi][yi][xi][outDof] = (outArr[zi][yi][xi + 1][inDof] - outArr[zi][yi][xi - 1][inDof])/4.0;
          }//end xi
        }//end yi
      }//end zi
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
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int yi = ys; yi < (ys + ny); ++yi, ++cnt) {
              int dof = (((K*(K + 1)) + K)*(K + 1)) + K - 1;
              first[cnt] = outArr[zi][yi][xs + 1][dof];
            }//end yi
          }//end zi
          MPI_Isend(&(first[0]), len, MPI_DOUBLE, prevX, 3, comm, &sReq3);
        }
      }
      if((xs + nx) == (Nx - 1)) {
        if(nx == 1) {
          MPI_Isend(&(L1[0]), len, MPI_DOUBLE, nextX, 4, comm, &sReq4);
        } else {
          for(int zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
            for(int yi = ys; yi < (ys + ny); ++yi, ++cnt) {
              int dof = (((K*(K + 1)) + K)*(K + 1)) + K - 1;
              last[cnt] = outArr[zi][yi][xs + nx - 2][dof];
            }//end yi
          }//end zi
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
      for(int zi = zs, gDof = 0; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi, ++gDof) {
          int outDof = (((K*(K + 1)) + K)*(K + 1)) + K;
          int inDof = (((K*(K + 1)) + K)*(K + 1)) + K - 1;
          if(ri == 0) {
            if(nx == 1) {
              outArr[zi][yi][xs][outDof] = -((3.0*outArr[zi][yi][xs][inDof]) - (4.0*R1[gDof]) + R2[gDof])/4.0;
            } else if(nx == 2) {
              outArr[zi][yi][xs][outDof] = -((3.0*outArr[zi][yi][xs][inDof]) -
                  (4.0*outArr[zi][yi][xs + 1][inDof]) + R1[gDof])/4.0;
              outArr[zi][yi][xs + 1][outDof] = (R1[gDof] - outArr[zi][yi][xs][inDof])/4.0;
            } else {
              outArr[zi][yi][xs][outDof] = -((3.0*outArr[zi][yi][xs][inDof]) - (4.0*outArr[zi][yi][xs + 1][inDof])
                  + outArr[zi][yi][xs + 2][inDof])/4.0;
              if(px == 1) {
                outArr[zi][yi][xs + nx - 1][outDof] = ((3.0*outArr[zi][yi][xs + nx - 1][inDof]) -
                    (4.0*outArr[zi][yi][xs + nx - 2][inDof]) + outArr[zi][yi][xs + nx - 3][inDof])/4.0;
              } else {
                outArr[zi][yi][xs + nx - 1][outDof] = (R1[gDof] - outArr[zi][yi][xs + nx - 2][inDof])/4.0;
              }
            }
          } else if(ri == (px - 1)) {
            if(nx == 1) {
              outArr[zi][yi][xs][outDof] = ((3.0*outArr[zi][yi][xs][inDof]) - (4.0*L1[gDof]) + L2[gDof])/4.0;
            } else if(nx == 2) {
              outArr[zi][yi][xs][outDof] = (outArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
              outArr[zi][yi][xs + 1][outDof] = ((3.0*outArr[zi][yi][xs + 1][inDof]) -
                  (4.0*outArr[zi][yi][xs][inDof]) + L1[gDof])/4.0;
            } else {
              outArr[zi][yi][xs][outDof] = (outArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
              outArr[zi][yi][xs + nx - 1][outDof] = ((3.0*outArr[zi][yi][xs + nx - 1][inDof]) -
                  (4.0*outArr[zi][yi][xs + nx - 2][inDof]) + outArr[zi][yi][xs + nx - 3][inDof])/4.0;
            }
          } else {
            if(nx == 1) {
              outArr[zi][yi][xs][outDof] = (R1[gDof] - L1[gDof])/4.0;
            } else {
              outArr[zi][yi][xs][outDof] = (outArr[zi][yi][xs + 1][inDof] - L1[gDof])/4.0;
              outArr[zi][yi][xs + nx - 1][outDof] = (R1[gDof] - outArr[zi][yi][xs + nx - 2][inDof])/4.0;
            }
          }
        }//end yi
      }//end zi
    }

    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  }
}


