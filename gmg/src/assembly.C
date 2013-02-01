
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void assembleKmat(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<Mat>& Kmat, std::vector<DM>& da, int K, std::vector<long long int>& coeffs, 
    std::vector<unsigned long long int>& factorialsList, bool print) {
  int nlevels = Kmat.size();
  for(int lev = 0; lev < nlevels; ++lev, print = false) {
    if(Kmat[lev] != NULL) {
      std::vector<std::vector<long double> > elemMat;
      if(dim == 1) {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, print);
      } else if(dim == 2) {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        long double hy = 1.0L/(static_cast<long double>(Ny[lev] - 1));
        createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, print);
      } else {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        long double hy = 1.0L/(static_cast<long double>(Ny[lev] - 1));
        long double hz = 1.0L/(static_cast<long double>(Nz[lev] - 1));
        createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMat, print);
      }
      computeKmat(Kmat[lev], da[lev], elemMat);
    }
  }//end lev
}

void assembleBlkKmats(std::vector<std::vector<std::vector<Mat> > >& blkKmats, int dim, int dofsPerNode,
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<DM>& da, int K, 
    std::vector<long long int>& coeffs, std::vector<unsigned long long int>& factorialsList) {
  int nlevels = da.size();
  for(int lev = 1; lev < nlevels; ++lev) {
    if(da[lev] != NULL) {
      std::vector<std::vector<long double> > elemMat;
      if(dim == 1) {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, false);
        for(int rDof = 0; rDof < dofsPerNode; ++rDof) {
          for(int cDof = rDof; cDof < dofsPerNode; ++cDof) {
            computeBlkKmat1D(blkKmats[lev - 1][rDof][cDof - rDof], da[lev], offsets[lev], elemMat, rDof, cDof);
          }//end cDof
        }//end rDof
      } else if(dim == 2) {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        long double hy = 1.0L/(static_cast<long double>(Ny[lev] - 1));
        createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, false);
        for(int rDof = 0; rDof < dofsPerNode; ++rDof) {
          for(int cDof = rDof; cDof < dofsPerNode; ++cDof) {
            computeBlkKmat2D(blkKmats[lev - 1][rDof][cDof - rDof], da[lev], partX[lev], offsets[lev],
                elemMat, rDof, cDof);
          }//end cDof
        }//end rDof
      } else {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        long double hy = 1.0L/(static_cast<long double>(Ny[lev] - 1));
        long double hz = 1.0L/(static_cast<long double>(Nz[lev] - 1));
        createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMat, false);
        for(int rDof = 0; rDof < dofsPerNode; ++rDof) {
          for(int cDof = rDof; cDof < dofsPerNode; ++cDof) {
            computeBlkKmat3D(blkKmats[lev - 1][rDof][cDof - rDof], da[lev], partY[lev], partX[lev], 
                offsets[lev], elemMat, rDof, cDof);
          }//end cDof
        }//end rDof
      }
    }
  }//end lev
}

void buildKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, bool print) {
  int nlevels = da.size();
  Kmat.resize(nlevels, NULL);
  for(int lev = 0; lev < nlevels; ++lev) {
    if(da[lev] != NULL) {
      DMCreateMatrix(da[lev], MATAIJ, &(Kmat[lev]));
      PetscInt sz;
      MatGetSize(Kmat[lev], &sz, PETSC_NULL);
      if(print) {
        std::cout<<"Lev = "<<lev<<", Kmat size = "<<sz<<std::endl;
      }
    }
  }//end lev
}

void buildBlkKmats(std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<DM>& da,
    std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes) {
  int nlevels = da.size();
  blkKmats.resize(nlevels - 1);
  for(int lev = 1; lev < nlevels; ++lev) {
    if(da[lev] != NULL) {
      PetscInt xs, ys, zs;
      PetscInt nx, ny, nz;
      DMDAGetCorners(da[lev], &xs, &ys, &zs, &nx, &ny, &nz);
      PetscInt dim;
      PetscInt dofsPerNode;
      PetscInt Nx;
      PetscInt Ny;
      PetscInt Nz;
      DMDAGetInfo(da[lev], &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
          &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);
      if(dim < 2) {
        Ny = 1;
        ny = 1;
        ys = 0;
      }
      if(dim < 3) {
        Nz = 1;
        nz = 1;
        zs = 0;
      }
      PetscInt locSz = (nx*ny*nz);
      PetscInt* d_nnz = new PetscInt[locSz];
      PetscInt* o_nnz = NULL;
      if(activeNpes[lev] > 1) {
        o_nnz = new PetscInt[locSz];
      }
      for(PetscInt zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          for(PetscInt xi = xs; xi < (xs + nx); ++xi, ++cnt) {
            d_nnz[cnt] = 0;
            if(o_nnz) {
              o_nnz[cnt] = 0;
            }
            for(int kk = -1; kk < 2; ++kk) {
              PetscInt oz = zi + kk;
              if((oz >= 0) && (oz < Nz)) {
                for(int jj = -1; jj < 2; ++jj) {
                  PetscInt oy = yi + jj;
                  if((oy >= 0) && (oy < Ny)) {
                    for(int ii = -1; ii < 2; ++ii) {
                      PetscInt ox = xi + ii;
                      if((ox >= 0) && (ox < Nx)) {
                        if((oz >= zs) && (oz < (zs + nz)) &&  
                            (oy >= ys) && (oy < (ys + ny)) &&
                            (ox >= xs) && (ox < (xs + nx))) {
                          ++(d_nnz[cnt]);
                        } else {
                          ++(o_nnz[cnt]);
                        }                
                      }
                    }//end ii
                  }
                }//end jj
              }
            }//end kk
          }//end xi
        }//end yi
      }//end zi
      blkKmats[lev - 1].resize(dofsPerNode);
      for(int di = 0; di < dofsPerNode; ++di) {
        blkKmats[lev - 1][di].resize((dofsPerNode - di), NULL);
        for(int dj = 0; dj < (dofsPerNode - di); ++dj) {
          MatCreate(activeComms[lev], &(blkKmats[lev - 1][di][dj]));
          MatSetSizes(blkKmats[lev - 1][di][dj], locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE);
          MatSetType(blkKmats[lev - 1][di][dj], MATAIJ);
          if(activeNpes[lev] > 1) {
            MatMPIAIJSetPreallocation(blkKmats[lev - 1][di][dj], -1, d_nnz, -1, o_nnz);
          } else {
            MatSeqAIJSetPreallocation(blkKmats[lev - 1][di][dj], -1, d_nnz);
          }
        }//end dj
      }//end di
      delete [] d_nnz;
      if(activeNpes[lev] > 1) {
        delete [] o_nnz;
      }
    }
  }//end lev
}

void computeKmat(Mat Kmat, DM da, std::vector<std::vector<long double> >& elemMat) {
  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DMDAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  PetscInt nxe = nx;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  PetscInt nye = ny;
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  }
  PetscInt nze = nz;
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  }

  MatZeroEntries(Kmat);

  if(dim == 1) {
    for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
      for(int rNode = 0, r = 0; rNode < 2; ++rNode) {
        for(int rDof = 0; rDof < dofsPerNode; ++rDof, ++r) {
          MatStencil row;
          row.i = xi + rNode;
          row.c = rDof;
          for(int cNode = 0, c = 0; cNode < 2; ++cNode) {
            for(int cDof = 0; cDof < dofsPerNode; ++cDof, ++c) {
              MatStencil col;
              col.i = xi + cNode;
              col.c = cDof;
              PetscScalar val = elemMat[r][c];
              MatSetValuesStencil(Kmat, 1, &row, 1, &col, &val, ADD_VALUES);
            }//end cDof
          }//end cNode
        }//end rDof
      }//end rNode
    }//end xi
  } else if(dim == 2) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        for(int rNodeY = 0, r = 0; rNodeY < 2; ++rNodeY) {
          for(int rNodeX = 0; rNodeX < 2; ++rNodeX) {
            for(int rDof = 0; rDof < dofsPerNode; ++rDof, ++r) {
              MatStencil row;
              row.j = yi + rNodeY;
              row.i = xi + rNodeX;
              row.c = rDof;
              for(int cNodeY = 0, c = 0; cNodeY < 2; ++cNodeY) {
                for(int cNodeX = 0; cNodeX < 2; ++cNodeX) {
                  for(int cDof = 0; cDof < dofsPerNode; ++cDof, ++c) {
                    MatStencil col;
                    col.j = yi + cNodeY;
                    col.i = xi + cNodeX;
                    col.c = cDof;
                    PetscScalar val = elemMat[r][c];
                    MatSetValuesStencil(Kmat, 1, &row, 1, &col, &val, ADD_VALUES);
                  }//end cDof
                }//end cNodeX
              }//end cNodeY
            }//end rDof
          }//end rNodeX
        }//end rNodeY
      }//end xi
    }//end yi
  } else {
    for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
      for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
        for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
          for(int rNodeZ = 0, r = 0; rNodeZ < 2; ++rNodeZ) {
            for(int rNodeY = 0; rNodeY < 2; ++rNodeY) {
              for(int rNodeX = 0; rNodeX < 2; ++rNodeX) {
                for(int rDof = 0; rDof < dofsPerNode; ++rDof, ++r) {
                  MatStencil row;
                  row.k = zi + rNodeZ;
                  row.j = yi + rNodeY;
                  row.i = xi + rNodeX;
                  row.c = rDof;
                  for(int cNodeZ = 0, c = 0; cNodeZ < 2; ++cNodeZ) {
                    for(int cNodeY = 0; cNodeY < 2; ++cNodeY) {
                      for(int cNodeX = 0; cNodeX < 2; ++cNodeX) {
                        for(int cDof = 0; cDof < dofsPerNode; ++cDof, ++c) {
                          MatStencil col;
                          col.k = zi + cNodeZ;
                          col.j = yi + cNodeY;
                          col.i = xi + cNodeX;
                          col.c = cDof;
                          PetscScalar val = elemMat[r][c];
                          MatSetValuesStencil(Kmat, 1, &row, 1, &col, &val, ADD_VALUES);
                        }//end cDof
                      }//end cNodeX
                    }//end cNodeY
                  }//end cNodeZ
                }//end rDof
              }//end rNodeX
            }//end rNodeY
          }//end rNodeZ
        }//end xi
      }//end yi
    }//end zi
  }

  MatAssemblyBegin(Kmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FINAL_ASSEMBLY);
}

void computeBlkKmat1D(Mat blkKmat, DM da, std::vector<PetscInt>& offsets,
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof) {
  PetscInt dofsPerNode;
  PetscInt Nx;
  DMDAGetInfo(da, PETSC_NULL, &Nx, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt nx;
  DMDAGetCorners(da, &xs, PETSC_NULL, PETSC_NULL, &nx, PETSC_NULL, PETSC_NULL);

  PetscInt nxe = nx;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ri = rank;

  MatZeroEntries(blkKmat);

  for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
    std::vector<PetscInt> indices(2);
    for(PetscInt x = 0; x < 2; ++x) {
      PetscInt vi = (xi + x);
      int pi = ri;
      PetscInt vXs = xs;
      if(vi >= (xs + nx)) {
        ++pi;
        vXs += nx;
      }
      PetscInt loc = vi - vXs;
      indices[x] = offsets[pi] + loc;
    }//end x
    for(int rNd = 0; rNd < 2; ++rNd) {
      int r = (rNd*dofsPerNode) + rDof;
      for(int cNd = 0; cNd < 2; ++cNd) {
        int c = (cNd*dofsPerNode) + cDof;
        PetscScalar val = elemMat[r][c];
        MatSetValues(blkKmat, 1, &(indices[rNd]), 1, &(indices[cNd]), &val, ADD_VALUES);
      }//end cNd
    }//end rNd
  }//end xi

  MatAssemblyBegin(blkKmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(blkKmat, MAT_FINAL_ASSEMBLY);
}

void computeBlkKmat2D(Mat blkKmat, DM da, std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets,
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof) {
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  DMDAGetInfo(da, PETSC_NULL, &Nx, &Ny, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt nx;
  PetscInt ny;
  DMDAGetCorners(da, &xs, &ys, PETSC_NULL, &nx, &ny, PETSC_NULL);

  PetscInt nxe = nx;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  PetscInt nye = ny;
  if((ys + ny) == Ny) {
    nye = ny - 1;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = partX.size();

  int rj = rank/px;
  int ri = rank%px;

  MatZeroEntries(blkKmat);

  for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
    for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
      std::vector<PetscInt> indices(4);
      for(PetscInt y = 0, v = 0; y < 2; ++y) {
        PetscInt vj = (yi + y);
        int pj = rj;
        PetscInt vYs = ys;
        if(vj >= (ys + ny)) {
          ++pj;
          vYs += ny;
        }
        PetscInt yLoc = vj - vYs;
        for(PetscInt x = 0; x < 2; ++x, ++v) {
          PetscInt vi = (xi + x);
          int pi = ri;
          PetscInt vXs = xs;
          if(vi >= (xs + nx)) {
            ++pi;
            vXs += nx;
          }
          PetscInt xLoc = vi - vXs;
          int pid = (pj*px) + pi;
          PetscInt loc = (yLoc*partX[pi]) + xLoc;
          indices[v] = offsets[pid] + loc;
        }//end x
      }//end y
      for(int rNd = 0; rNd < 4; ++rNd) {
        int r = (rNd*dofsPerNode) + rDof;
        for(int cNd = 0; cNd < 4; ++cNd) {
          int c = (cNd*dofsPerNode) + cDof;
          PetscScalar val = elemMat[r][c];
          MatSetValues(blkKmat, 1, &(indices[rNd]), 1, &(indices[cNd]), &val, ADD_VALUES);
        }//end cNd
      }//end rNd
    }//end xi
  }//end yi

  MatAssemblyBegin(blkKmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(blkKmat, MAT_FINAL_ASSEMBLY);
}

void computeBlkKmat3D(Mat blkKmat, DM da, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, 
    std::vector<std::vector<long double> >& elemMat, int rDof, int cDof) {
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DMDAGetInfo(da, PETSC_NULL, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  PetscInt nxe = nx;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  PetscInt nye = ny;
  if((ys + ny) == Ny) {
    nye = ny - 1;
  }
  PetscInt nze = nz;
  if((zs + nz) == Nz) {
    nze = nz - 1;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = partX.size();
  int py = partY.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  MatZeroEntries(blkKmat);

  for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        std::vector<PetscInt> indices(8);
        for(PetscInt z = 0, v = 0; z < 2; ++z) {
          PetscInt vk = (zi + z);
          int pk = rk;
          PetscInt vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          PetscInt zLoc = vk - vZs;
          for(PetscInt y = 0; y < 2; ++y) {
            PetscInt vj = (yi + y);
            int pj = rj;
            PetscInt vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            PetscInt yLoc = vj - vYs;
            for(PetscInt x = 0; x < 2; ++x, ++v) {
              PetscInt vi = (xi + x);
              int pi = ri;
              PetscInt vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              PetscInt xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              PetscInt loc = (((zLoc*partY[pj]) + yLoc)*partX[pi]) + xLoc;
              indices[v] = offsets[pid] + loc;
            }//end x
          }//end y
        }//end z
        for(int rNd = 0; rNd < 8; ++rNd) {
          int r = (rNd*dofsPerNode) + rDof;
          for(int cNd = 0; cNd < 8; ++cNd) {
            int c = (cNd*dofsPerNode) + cDof;
            PetscScalar val = elemMat[r][c];
            MatSetValues(blkKmat, 1, &(indices[rNd]), 1, &(indices[cNd]), &val, ADD_VALUES);
          }//end cNd
        }//end rNd
      }//end xi
    }//end yi
  }//end zi

  MatAssemblyBegin(blkKmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(blkKmat, MAT_FINAL_ASSEMBLY);
}

