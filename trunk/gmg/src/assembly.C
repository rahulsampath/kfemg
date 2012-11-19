
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

extern PetscLogEvent buildKmatEvent;
extern PetscLogEvent buildKblkDiagEvent;
extern PetscLogEvent buildKblkUpperEvent;

void buildKupperBlocks(std::vector<unsigned long long int>& factorialsList,
    std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats) {
  PetscLogEventBegin(buildKblkUpperEvent, 0, 0, 0, 0);

  int factor = 3;
  if(dim > 1) {
    factor *= 3;
  }
  if(dim > 2) {
    factor *= 3;
  }
  Kblk.resize(da.size());
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt nx, ny, nz;
      DMDAGetCorners(da[i], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nx, &ny, &nz);
      if(dim < 2) {
        ny = 1;
      }
      if(dim < 3) {
        nz = 1;
      }
      PetscInt locSz = (nx*ny*nz);
      Kblk[i].resize((dofsPerNode - 1), NULL);
      for(int d = 0; d < (dofsPerNode - 1); ++d) {
        MatCreate(activeComms[i], &(Kblk[i][d]));
        MatSetSizes(Kblk[i][d], locSz, (locSz*(dofsPerNode - d - 1)), PETSC_DETERMINE, PETSC_DETERMINE);
        MatSetType(Kblk[i][d], MATAIJ);
        if(activeNpes[i] > 1) {
          MatMPIAIJSetPreallocation(Kblk[i][d], (factor*(dofsPerNode - d - 1)), PETSC_NULL,
              ((factor - 1)*(dofsPerNode - d - 1)), PETSC_NULL);
        } else {
          MatSeqAIJSetPreallocation(Kblk[i][d], (factor*(dofsPerNode - d - 1)), PETSC_NULL);
        }
        computeKblkUpper(factorialsList, Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i], elemMats[i], coeffs, K, d);
        if(d == 0) {
          dirichletMatrixCorrectionBlkUpper(Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i]);
        }
      }//end d
    }
  }//end i

  PetscLogEventEnd(buildKblkUpperEvent, 0, 0, 0, 0);
}

void buildKdiagBlocks(std::vector<unsigned long long int>& factorialsList,
    std::vector<std::vector<Mat> >& Kblk, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats) {
  PetscLogEventBegin(buildKblkDiagEvent, 0, 0, 0, 0);

  int factor = 3;
  if(dim > 1) {
    factor *= 3;
  }
  if(dim > 2) {
    factor *= 3;
  }
  Kblk.resize(da.size());
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt nx, ny, nz;
      DMDAGetCorners(da[i], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nx, &ny, &nz);
      if(dim < 2) {
        ny = 1;
      }
      if(dim < 3) {
        nz = 1;
      }
      PetscInt locSz = (nx*ny*nz);
      Kblk[i].resize(dofsPerNode, NULL);
      for(int d = 0; d < dofsPerNode; ++d) {
        MatCreate(activeComms[i], &(Kblk[i][d]));
        MatSetSizes(Kblk[i][d], locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE);
        MatSetType(Kblk[i][d], MATAIJ);
        if(activeNpes[i] > 1) {
          MatMPIAIJSetPreallocation(Kblk[i][d], factor, PETSC_NULL, (factor - 1), PETSC_NULL);
        } else {
          MatSeqAIJSetPreallocation(Kblk[i][d], factor, PETSC_NULL);
        }
        computeKblkDiag(factorialsList, Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i], elemMats[i], coeffs, K, d);
        if(d == 0) {
          dirichletMatrixCorrectionBlkDiag(Kblk[i][d], da[i], lz[i], ly[i], lx[i], offsets[i]);
        }
      }//end d
    }
  }//end i

  PetscLogEventEnd(buildKblkDiagEvent, 0, 0, 0, 0);
}

void buildKmat(std::vector<unsigned long long int>& factorialsList,
    std::vector<Mat>& Kmat, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets, std::vector<std::vector<std::vector<long double> > >& elemMats, bool print) {
  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  int factor = 3;
  if(dim > 1) {
    factor *= 3;
  }
  if(dim > 2) {
    factor *= 3;
  }
  Kmat.resize(da.size(), NULL);
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt nx, ny, nz;
      DMDAGetCorners(da[i], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nx, &ny, &nz);
      if(dim < 2) {
        ny = 1;
      }
      if(dim < 3) {
        nz = 1;
      }
      PetscInt locSz = (nx*ny*nz*dofsPerNode);
      MatCreate(activeComms[i], &(Kmat[i]));
      MatSetSizes(Kmat[i], locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE);
      MatSetType(Kmat[i], MATAIJ);
      if(activeNpes[i] > 1) {
        MatMPIAIJSetPreallocation(Kmat[i], (factor*dofsPerNode), PETSC_NULL, ((factor - 1)*dofsPerNode), PETSC_NULL);
      } else {
        MatSeqAIJSetPreallocation(Kmat[i], (factor*dofsPerNode), PETSC_NULL);
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
      computeKmat(factorialsList, Kmat[i], da[i], lz[i], ly[i], lx[i], offsets[i], elemMats[i], coeffs, K, printInt);
      dirichletMatrixCorrection(Kmat[i], da[i], lz[i], ly[i], lx[i], offsets[i]);
    }
    if(print) {
      std::cout<<"Built Kmat for level = "<<i<<std::endl;
    }
  }//end i

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);
}

void computeKmat(std::vector<unsigned long long int>& factorialsList,
    Mat Kmat, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<std::vector<long double> >& elemMat, 
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

  int numZnodes = 2;
  int numYnodes = 2;
  int numXnodes = 2;

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
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  MatZeroEntries(Kmat);

  for(unsigned int zi = zs; zi < (zs + nze); ++zi) {
    for(unsigned int yi = ys; yi < (ys + nye); ++yi) {
      for(unsigned int xi = xs; xi < (xs + nxe); ++xi) {
        for(int z = 0, i = 0; z < numZnodes; ++z) {
          int vk = (zi + z);
          int pk = rk;
          int vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          int zLoc = vk - vZs;
          for(int y = 0; y < numYnodes; ++y) {
            int vj = (yi + y);
            int pj = rj;
            int vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            int yLoc = vj - vYs;
            for(int x = 0; x < numXnodes; ++x) {
              int vi = (xi + x);
              int pi = ri;
              int vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              int xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              int loc = (((zLoc*ly[pj]) + yLoc)*lx[pi]) + xLoc;
              int idBase = ((offsets[pid] + loc)*dofsPerNode);
              for(unsigned int d = 0; d < dofsPerNode; ++i, ++d) {
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

void computeKblkDiag(std::vector<unsigned long long int>& factorialsList,
    Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<std::vector<long double> >& elemMat, 
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

  int numZnodes = 2;
  int numYnodes = 2;
  int numXnodes = 2;

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
  for(int r = 0, i = 0; r < nodesPerElem; ++r) {
    for(int c = 0; c < nodesPerElem; ++c, ++i) {
      vals[i] = elemMat[(r*dofsPerNode) + dof][(c*dofsPerNode) + dof];
    }//end c
  }//end r

  std::vector<PetscInt> indices(nodesPerElem);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  MatZeroEntries(Kblk);

  for(unsigned int zi = zs; zi < (zs + nze); ++zi) {
    for(unsigned int yi = ys; yi < (ys + nye); ++yi) {
      for(unsigned int xi = xs; xi < (xs + nxe); ++xi) {
        for(int z = 0, i = 0; z < numZnodes; ++z) {
          int vk = (zi + z);
          int pk = rk;
          int vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          int zLoc = vk - vZs;
          for(int y = 0; y < numYnodes; ++y) {
            int vj = (yi + y);
            int pj = rj;
            int vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            int yLoc = vj - vYs;
            for(int x = 0; x < numXnodes; ++x, ++i) {
              int vi = (xi + x);
              int pi = ri;
              int vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              int xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              int loc = (((zLoc*ly[pj]) + yLoc)*lx[pi]) + xLoc;
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

void computeKblkUpper(std::vector<unsigned long long int>& factorialsList,
    Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<std::vector<long double> >& elemMat, 
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

  int numZnodes = 2;
  int numYnodes = 2;
  int numXnodes = 2;

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
  unsigned int colIdFactor = dofsPerNode - dof - 1;

  std::vector<PetscInt> rIndices(nodesPerElem);
  std::vector<PetscInt> cIndices(nodesPerElem*colIdFactor);

  std::vector<PetscScalar> vals((rIndices.size())*(cIndices.size()));
  for(size_t r = 0, i = 0; r < nodesPerElem; ++r) {
    for(size_t c = 0; c < nodesPerElem; ++c) {
      for(size_t d = (dof + 1); d < dofsPerNode; ++d, ++i) {
        vals[i] = elemMat[(r*dofsPerNode) + dof][(c*dofsPerNode) + d];
      }//end d
    }//end c
  }//end r

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  MatZeroEntries(Kblk);

  for(unsigned int zi = zs; zi < (zs + nze); ++zi) {
    for(unsigned int yi = ys; yi < (ys + nye); ++yi) {
      for(unsigned int xi = xs; xi < (xs + nxe); ++xi) {
        for(int z = 0, r = 0; z < numZnodes; ++z) {
          int vk = (zi + z);
          int pk = rk;
          int vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          int zLoc = vk - vZs;
          for(int y = 0; y < numYnodes; ++y) {
            int vj = (yi + y);
            int pj = rj;
            int vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            int yLoc = vj - vYs;
            for(int x = 0; x < numXnodes; ++x, ++r) {
              int vi = (xi + x);
              int pi = ri;
              int vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              int xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              int loc = (((zLoc*ly[pj]) + yLoc)*lx[pi]) + xLoc;
              rIndices[r] = offsets[pid] + loc;
            }//end x
          }//end y
        }//end z
        for(int r = 0, c = 0; r < nodesPerElem; ++r) {
          int idBase = (rIndices[r])*colIdFactor;
          for(unsigned int d = 0; d < colIdFactor; ++c, ++d) {
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
  for(int i = 1; i < elemMats.size(); ++i) {
    elemMats[i].resize(elemMats[i-1].size());
    for(int j = 0; j < elemMats[i].size(); ++j) {
      elemMats[i][j].resize(elemMats[i-1][j].size());
      for(int k = 0; k < elemMats[i][j].size(); ++k) {
        elemMats[i][j][k] = scaling*elemMats[i - 1][j][k];
      }//end k
    }//end j
  }//end i
}




