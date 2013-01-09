
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

extern PetscLogEvent buildKmatEvent;
extern PetscLogEvent buildKblkDiagEvent;
extern PetscLogEvent buildKblkUpperEvent;

/*
void buildKupperBlocks(std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats) {
  PetscLogEventBegin(buildKblkUpperEvent, 0, 0, 0, 0);

  Kblk.resize(da.size());
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt xs, ys, zs;
      PetscInt nx, ny, nz;
      DMDAGetCorners(da[i], &xs, &ys, &zs, &nx, &ny, &nz);
      PetscInt dim;
      PetscInt dofsPerNode;
      PetscInt Nx;
      PetscInt Ny;
      PetscInt Nz;
      DMDAGetInfo(da[i], &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
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
      if(activeNpes[i] > 1) {
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
      Kblk[i].resize((dofsPerNode - 1), NULL);
      for(PetscInt d = 0; d < (dofsPerNode - 1); ++d) {
        PetscInt colFac = dofsPerNode - d - 1;
        for(PetscInt j = 0; j < locSz; ++j) {
          d_nnz[j] *= colFac;
          if(o_nnz) {
            o_nnz[j] *= colFac;
          }
        }//end j
        MatCreate(activeComms[i], &(Kblk[i][d]));
        MatSetSizes(Kblk[i][d], locSz, (locSz*colFac), PETSC_DETERMINE, PETSC_DETERMINE);
        MatSetType(Kblk[i][d], MATAIJ);
        if(activeNpes[i] > 1) {
          MatMPIAIJSetPreallocation(Kblk[i][d], -1, d_nnz, -1, o_nnz);
        } else {
          MatSeqAIJSetPreallocation(Kblk[i][d], -1, d_nnz);
        }
        for(PetscInt j = 0; j < locSz; ++j) {
          d_nnz[j] /= colFac;
          if(o_nnz) {
            o_nnz[j] /= colFac;
          }
        }//end j
      }//end d
      delete [] d_nnz;
      if(activeNpes[i] > 1) {
        delete [] o_nnz;
      }
      for(PetscInt d = 0; d < (dofsPerNode - 1); ++d) {
        computeKblkUpper(Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i], elemMats[i], coeffs, K, d);
        if(d == 0) {
          dirichletMatrixCorrectionBlkUpper(Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i]);
        }
      }//end d
    }
  }//end i

  PetscLogEventEnd(buildKblkUpperEvent, 0, 0, 0, 0);
}

void buildKdiagBlocks(std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats) {
  PetscLogEventBegin(buildKblkDiagEvent, 0, 0, 0, 0);

  Kblk.resize(da.size());
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt xs, ys, zs;
      PetscInt nx, ny, nz;
      DMDAGetCorners(da[i], &xs, &ys, &zs, &nx, &ny, &nz);
      PetscInt dim;
      PetscInt dofsPerNode;
      PetscInt Nx;
      PetscInt Ny;
      PetscInt Nz;
      DMDAGetInfo(da[i], &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
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
      if(activeNpes[i] > 1) {
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
      Kblk[i].resize(dofsPerNode, NULL);
      for(PetscInt d = 0; d < dofsPerNode; ++d) {
        MatCreate(activeComms[i], &(Kblk[i][d]));
        MatSetSizes(Kblk[i][d], locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE);
        MatSetType(Kblk[i][d], MATAIJ);
        if(activeNpes[i] > 1) {
          MatMPIAIJSetPreallocation(Kblk[i][d], -1, d_nnz, -1, o_nnz);
        } else {
          MatSeqAIJSetPreallocation(Kblk[i][d], -1, d_nnz);
        }
      }//end d
      delete [] d_nnz;
      if(activeNpes[i] > 1) {
        delete [] o_nnz;
      }
      for(PetscInt d = 0; d < dofsPerNode; ++d) {
        computeKblkDiag(Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i], elemMats[i], coeffs, K, d);
        if(d == 0) {
          dirichletMatrixCorrectionBlkDiag(Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i]);
        }
      }//end d
    }
  }//end i

  PetscLogEventEnd(buildKblkDiagEvent, 0, 0, 0, 0);
}
*/

void buildKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<PetscInt> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats, bool print) {
  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  Kmat.resize(da.size(), NULL);
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt xs, ys, zs;
      PetscInt nx, ny, nz;
      DMDAGetCorners(da[i], &xs, &ys, &zs, &nx, &ny, &nz);
      PetscInt dim;
      PetscInt dofsPerNode;
      PetscInt Nx;
      PetscInt Ny;
      PetscInt Nz;
      DMDAGetInfo(da[i], &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
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
      PetscInt locSz = (nx*ny*nz*dofsPerNode);
      PetscInt* d_nnz = new PetscInt[locSz];
      PetscInt* o_nnz = NULL;
      if(activeNpes[i] > 1) {
        o_nnz = new PetscInt[locSz];
      }
      for(PetscInt zi = zs, cnt = 0; zi < (zs + nz); ++zi) {
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
            PetscInt diagVal = 0;
            PetscInt offVal = 0;
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
                          diagVal += dofsPerNode;
                        } else {
                          offVal += dofsPerNode;
                        }                
                      }
                    }//end ii
                  }
                }//end jj
              }
            }//end kk
            for(int d = 0; d < dofsPerNode; ++d, ++cnt) {
              d_nnz[cnt] = diagVal;
              if(o_nnz) {
                o_nnz[cnt] = offVal;
              }
            }//end d
          }//end xi
        }//end yi
      }//end zi
      MatCreate(activeComms[i], &(Kmat[i]));
      MatSetSizes(Kmat[i], locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE);
      MatSetType(Kmat[i], MATAIJ);
      if(activeNpes[i] > 1) {
        MatMPIAIJSetPreallocation(Kmat[i], -1, d_nnz, -1, o_nnz);
      } else {
        MatSeqAIJSetPreallocation(Kmat[i], -1, d_nnz);
      }
      delete [] d_nnz;
      if(activeNpes[i] > 1) {
        delete [] o_nnz;
      }
      PetscInt sz;
      MatGetSize(Kmat[i], &sz, PETSC_NULL);
      if(print) {
        std::cout<<"Kmat Size for level "<<i<<" = "<<sz<<std::endl;
      }
      bool printInt = print;
      if(i > 0) {
        printInt = false;
      }
      computeKmat(Kmat[i], da[i], lz[i], ly[i], lx[i], offsets[i], elemMats[i], coeffs, K, printInt);
      dirichletMatrixCorrection(Kmat[i], da[i], lz[i], ly[i], lx[i], offsets[i]);
    }
    if(print) {
      std::cout<<"Built Kmat for level = "<<i<<std::endl;
    }
  }//end i

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);
}

void computeKmat(Mat Kmat, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, 
    std::vector<long long int>& coeffs, const unsigned int K, bool print) {
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

#ifdef DEBUG
  if(dim < 2) {
    assert(Ny == 1);
    assert(ys == 0);
    assert(ny == 1);
  }
  if(dim < 3) {
    assert(Nz == 1);
    assert(zs == 0);
    assert(nz == 1);
  }
#endif

  PetscInt nxe = nx;
  PetscInt nye = ny;
  PetscInt nze = nz;

  PetscInt numZnodes = 2;
  PetscInt numYnodes = 2;
  PetscInt numXnodes = 2;

  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  } else {
    numYnodes = 1;
  }
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  } else {
    numZnodes = 1;
  }

  std::vector<PetscScalar> vals((elemMat.size())*(elemMat.size()));
  for(size_t r = 0, i = 0; r < (elemMat.size()); ++r) {
    for(size_t c = 0; c < (elemMat.size()); ++c, ++i) {
      vals[i] = elemMat[r][c];
    }//end c
  }//end r

  unsigned int nodesPerElem = (1 << dim);

  std::vector<PetscInt> indices(nodesPerElem*dofsPerNode);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  MatZeroEntries(Kmat);

  for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        for(PetscInt z = 0, i = 0; z < numZnodes; ++z) {
          PetscInt vk = (zi + z);
          int pk = rk;
          PetscInt vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          PetscInt zLoc = vk - vZs;
          for(PetscInt y = 0; y < numYnodes; ++y) {
            PetscInt vj = (yi + y);
            int pj = rj;
            PetscInt vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            PetscInt yLoc = vj - vYs;
            for(PetscInt x = 0; x < numXnodes; ++x) {
              PetscInt vi = (xi + x);
              int pi = ri;
              PetscInt vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              PetscInt xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              PetscInt loc = (((zLoc*ly[pj]) + yLoc)*lx[pi]) + xLoc;
              PetscInt idBase = ((offsets[pid] + loc)*dofsPerNode);
              for(PetscInt d = 0; d < dofsPerNode; ++i, ++d) {
                indices[i] = idBase + d;
              }//end d
            }//end x
          }//end y
        }//end z
        MatSetValues(Kmat, (indices.size()), &(indices[0]),
            (indices.size()), &(indices[0]), &(vals[0]), ADD_VALUES);
      }//end xi
    }//end yi
  }//end zi

  MatAssemblyBegin(Kmat, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FLUSH_ASSEMBLY);
}

/*
void computeKblkDiag(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, 
    std::vector<long long int>& coeffs, const unsigned int K, const unsigned int dof) {
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

#ifdef DEBUG
  if(dim < 2) {
    assert(Ny == 1);
    assert(ys == 0);
    assert(ny == 1);
  }
  if(dim < 3) {
    assert(Nz == 1);
    assert(zs == 0);
    assert(nz == 1);
  }
#endif

  PetscInt nxe = nx;
  PetscInt nye = ny;
  PetscInt nze = nz;

  PetscInt numZnodes = 2;
  PetscInt numYnodes = 2;
  PetscInt numXnodes = 2;

  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  } else {
    numYnodes = 1;
  }
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  } else {
    numZnodes = 1;
  }

  unsigned int nodesPerElem = (1 << dim);

  std::vector<PetscScalar> vals(nodesPerElem*nodesPerElem);
  for(unsigned int r = 0, i = 0; r < nodesPerElem; ++r) {
    for(unsigned int c = 0; c < nodesPerElem; ++c, ++i) {
      vals[i] = elemMat[(r*dofsPerNode) + dof][(c*dofsPerNode) + dof];
    }//end c
  }//end r

  std::vector<PetscInt> indices(nodesPerElem);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  MatZeroEntries(Kblk);

  for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        for(PetscInt z = 0, i = 0; z < numZnodes; ++z) {
          PetscInt vk = (zi + z);
          int pk = rk;
          PetscInt vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          PetscInt zLoc = vk - vZs;
          for(PetscInt y = 0; y < numYnodes; ++y) {
            PetscInt vj = (yi + y);
            int pj = rj;
            PetscInt vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            PetscInt yLoc = vj - vYs;
            for(PetscInt x = 0; x < numXnodes; ++x, ++i) {
              PetscInt vi = (xi + x);
              int pi = ri;
              PetscInt vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              PetscInt xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              PetscInt loc = (((zLoc*ly[pj]) + yLoc)*lx[pi]) + xLoc;
              indices[i] = offsets[pid] + loc;
            }//end x
          }//end y
        }//end z
        MatSetValues(Kblk, (indices.size()), &(indices[0]),
            (indices.size()), &(indices[0]), &(vals[0]), ADD_VALUES);
      }//end xi
    }//end yi
  }//end zi

  if(dof == 0) {
    MatAssemblyBegin(Kblk, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(Kblk, MAT_FLUSH_ASSEMBLY);
  } else {
    MatAssemblyBegin(Kblk, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Kblk, MAT_FINAL_ASSEMBLY);
  }
}

void computeKblkUpper(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<std::vector<long double> >& elemMat, 
    std::vector<long long int>& coeffs, const unsigned int K, const unsigned int dof) {
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

#ifdef DEBUG
  if(dim < 2) {
    assert(Ny == 1);
    assert(ys == 0);
    assert(ny == 1);
  }
  if(dim < 3) {
    assert(Nz == 1);
    assert(zs == 0);
    assert(nz == 1);
  }
#endif

  PetscInt nxe = nx;
  PetscInt nye = ny;
  PetscInt nze = nz;

  PetscInt numZnodes = 2;
  PetscInt numYnodes = 2;
  PetscInt numXnodes = 2;

  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  } else {
    numYnodes = 1;
  }
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  } else {
    numZnodes = 1;
  }

  unsigned int nodesPerElem = (1 << dim);
  PetscInt colIdFactor = dofsPerNode - dof - 1;

  std::vector<PetscInt> rIndices(nodesPerElem);
  std::vector<PetscInt> cIndices(nodesPerElem*colIdFactor);

  std::vector<PetscScalar> vals((rIndices.size())*(cIndices.size()));
  for(unsigned int r = 0, i = 0; r < nodesPerElem; ++r) {
    for(unsigned int c = 0; c < nodesPerElem; ++c) {
      for(int d = (dof + 1); d < dofsPerNode; ++d, ++i) {
        vals[i] = elemMat[(r*dofsPerNode) + dof][(c*dofsPerNode) + d];
      }//end d
    }//end c
  }//end r

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  MatZeroEntries(Kblk);

  for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        for(PetscInt z = 0, r = 0; z < numZnodes; ++z) {
          PetscInt vk = (zi + z);
          int pk = rk;
          PetscInt vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          PetscInt zLoc = vk - vZs;
          for(PetscInt y = 0; y < numYnodes; ++y) {
            PetscInt vj = (yi + y);
            int pj = rj;
            PetscInt vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            PetscInt yLoc = vj - vYs;
            for(PetscInt x = 0; x < numXnodes; ++x, ++r) {
              PetscInt vi = (xi + x);
              int pi = ri;
              PetscInt vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              PetscInt xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              PetscInt loc = (((zLoc*ly[pj]) + yLoc)*lx[pi]) + xLoc;
              rIndices[r] = offsets[pid] + loc;
            }//end x
          }//end y
        }//end z
        for(unsigned int r = 0, c = 0; r < nodesPerElem; ++r) {
          PetscInt idBase = (rIndices[r])*colIdFactor;
          for(PetscInt d = 0; d < colIdFactor; ++c, ++d) {
            cIndices[c] = idBase + d;
          }//end d
        }//end r
        MatSetValues(Kblk, (rIndices.size()), &(rIndices[0]),
            (cIndices.size()), &(cIndices[0]), &(vals[0]), ADD_VALUES);
      }//end xi
    }//end yi
  }//end zi

  if(dof == 0) {
    MatAssemblyBegin(Kblk, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(Kblk, MAT_FLUSH_ASSEMBLY);
  } else {
    MatAssemblyBegin(Kblk, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Kblk, MAT_FINAL_ASSEMBLY);
  }
}
*/

void createElementMatrices(std::vector<unsigned long long int>& factorialsList, int dim, int K, 
    std::vector<long long int>& coeffs, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<std::vector<long double> > >& elemMats, bool print) {
  elemMats.resize(Nx.size());
  long double scaling;
  if(dim == 1) {
    long double hx = 1.0L/(static_cast<long double>(Nx[0] - 1));
    createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMats[0], print);
    scaling = 2.0;
  } else if(dim == 2) {
    long double hx = 1.0L/(static_cast<long double>(Nx[0] - 1));
    long double hy = 1.0L/(static_cast<long double>(Ny[0] - 1));
    createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMats[0], print);
    scaling = 1.0;
  } else {
    long double hx = 1.0L/(static_cast<long double>(Nx[0] - 1));
    long double hy = 1.0L/(static_cast<long double>(Ny[0] - 1));
    long double hz = 1.0L/(static_cast<long double>(Nz[0] - 1));
    createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMats[0], print);
    scaling = 0.5;
  }
  for(size_t i = 1; i < elemMats.size(); ++i) {
    elemMats[i].resize(elemMats[i-1].size());
    for(size_t j = 0; j < elemMats[i].size(); ++j) {
      elemMats[i][j].resize(elemMats[i-1][j].size());
      for(size_t k = 0; k < elemMats[i][j].size(); ++k) {
        elemMats[i][j][k] = scaling*elemMats[i - 1][j][k];
      }//end k
    }//end j
  }//end i
}




