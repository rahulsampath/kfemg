
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

extern PetscLogEvent createDAevent;
extern PetscLogEvent buildKmatEvent;
extern PetscLogEvent buildKblkDiagEvent;
extern PetscLogEvent buildKblkUpperEvent;
extern PetscLogEvent vCycleEvent;

void createPCShellData(std::vector<PCShellData>& data) {
}

void buildKdiagBlocks(std::vector<unsigned long long int>& factorialsList,
    std::vector<std::vector<Mat> >& Kblk, std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets) {
  PetscLogEventBegin(buildKblkDiagEvent, 0, 0, 0, 0);

  int factor = 3;
  if(dim > 1) {
    factor *= 3;
  }
  if(dim > 2) {
    factor *= 3;
  }
  Kblk.resize((da.size()) - 1);
  for(int i = 1; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt nx, ny, nz;
      DAGetCorners(da[i], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nx, &ny, &nz);
      if(dim < 2) {
        ny = 1;
      }
      if(dim < 3) {
        nz = 1;
      }
      PetscInt locSz = (nx*ny*nz);
      Kblk[i - 1].resize(dofsPerNode, NULL);
      for(int d = 0; d < dofsPerNode; ++d) {
        MatCreate(activeComms[i], &(Kblk[i - 1][d]));
        MatSetSizes(Kblk[i - 1][d], locSz, locSz, PETSC_DETERMINE, PETSC_DETERMINE);
        MatSetType(Kblk[i - 1][d], MATAIJ);
        if(activeNpes[i] > 1) {
          MatMPIAIJSetPreallocation(Kblk[i - 1][d], factor, PETSC_NULL, (factor - 1), PETSC_NULL);
        } else {
          MatSeqAIJSetPreallocation(Kblk[i - 1][d], factor, PETSC_NULL);
        }
        computeKblkDiag(factorialsList, Kblk[i - 1][d], da[i], lz[i], ly[i], lx[i], offsets[i], coeffs, K, d);
        if(d == 0) {
          dirichletMatrixCorrectionBlkDiag(Kblk[i - 1][d], da[i], lz[i], ly[i], lx[i], offsets[i]);
        }
      }//end d
    }
  }//end i

  PetscLogEventEnd(buildKblkDiagEvent, 0, 0, 0, 0);
}

void buildKupperBlocks(std::vector<unsigned long long int>& factorialsList,
    std::vector<std::vector<Mat> >& Kblk, std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets) {
  PetscLogEventBegin(buildKblkUpperEvent, 0, 0, 0, 0);

  int factor = 3;
  if(dim > 1) {
    factor *= 3;
  }
  if(dim > 2) {
    factor *= 3;
  }
  Kblk.resize((da.size()) - 1);
  for(int i = 1; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscInt nx, ny, nz;
      DAGetCorners(da[i], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nx, &ny, &nz);
      if(dim < 2) {
        ny = 1;
      }
      if(dim < 3) {
        nz = 1;
      }
      PetscInt locSz = (nx*ny*nz);
      Kblk[i - 1].resize(dofsPerNode, NULL);
      for(int d = 0; d < dofsPerNode; ++d) {
        MatCreate(activeComms[i], &(Kblk[i - 1][d]));
        MatSetSizes(Kblk[i - 1][d], locSz, (locSz*(dofsPerNode - d - 1)), PETSC_DETERMINE, PETSC_DETERMINE);
        MatSetType(Kblk[i - 1][d], MATAIJ);
        if(activeNpes[i] > 1) {
          MatMPIAIJSetPreallocation(Kblk[i - 1][d], (factor*(dofsPerNode - d - 1)), PETSC_NULL,
              ((factor - 1)*(dofsPerNode - d - 1)), PETSC_NULL);
        } else {
          MatSeqAIJSetPreallocation(Kblk[i - 1][d], (factor*(dofsPerNode - d - 1)), PETSC_NULL);
        }
        computeKblkUpper(factorialsList, Kblk[i - 1][d], da[i], lz[i], ly[i], lx[i], offsets[i], coeffs, K, d);
        if(d == 0) {
          dirichletMatrixCorrectionBlkUpper(Kblk[i - 1][d], da[i], lz[i], ly[i], lx[i], offsets[i]);
        }
      }//end d
    }
  }//end i

  PetscLogEventEnd(buildKblkUpperEvent, 0, 0, 0, 0);
}

void buildKmat(std::vector<unsigned long long int>& factorialsList,
    std::vector<Mat>& Kmat, std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets, bool print) {
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
      DAGetCorners(da[i], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nx, &ny, &nz);
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
      computeKmat(factorialsList, Kmat[i], da[i], lz[i], ly[i], lx[i], offsets[i], coeffs, K, printInt);
      dirichletMatrixCorrection(Kmat[i], da[i], lz[i], ly[i], lx[i], offsets[i]);
    }
    if(print) {
      std::cout<<"Built Kmat for level = "<<i<<std::endl;
    }
  }//end i

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);
}

void computeKmat(std::vector<unsigned long long int>& factorialsList,
    Mat Kmat, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<long long int>& coeffs, const unsigned int K, bool print) {
  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

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

  long double hx, hy, hz;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  hx = 1.0L/(static_cast<long double>(Nx - 1));
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  } else {
    numYnodes = 1;
  }
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  } else {
    numZnodes = 1;
  }

  std::vector<std::vector<long double> > elemMat;
  if(dim == 1) {
    createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, print);
  } else if(dim == 2) {
    createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, print);
  } else {
    createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMat, print);
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
    Mat Kblk, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<long long int>& coeffs, const unsigned int K, const unsigned int dof) {
  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

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

  long double hx, hy, hz;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  hx = 1.0L/(static_cast<long double>(Nx - 1));
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  } else {
    numYnodes = 1;
  }
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  } else {
    numZnodes = 1;
  }

  std::vector<std::vector<long double> > elemMat;
  if(dim == 1) {
    createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, false);
  } else if(dim == 2) {
    createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, false);
  } else {
    createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMat, false);
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
    Mat Kblk, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<long long int>& coeffs, const unsigned int K, const unsigned int dof) {
  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

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

  long double hx, hy, hz;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  hx = 1.0L/(static_cast<long double>(Nx - 1));
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  } else {
    numYnodes = 1;
  }
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  } else {
    numZnodes = 1;
  }

  std::vector<std::vector<long double> > elemMat;
  if(dim == 1) {
    createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, false);
  } else if(dim == 2) {
    createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, false);
  } else {
    createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMat, false);
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

void dirichletMatrixCorrection(Mat Kmat, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<int>& offsets) {
  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  std::vector<PetscInt> xvec;
  if(xs == 0) {
    xvec.push_back(0);
  }
  if((xs + nx) == Nx) {
    xvec.push_back((Nx - 1));
  }

  std::vector<PetscInt> yvec;
  if(dim > 1) {
    if(ys == 0) {
      yvec.push_back(0);
    }
    if((ys + ny) == Ny) {
      yvec.push_back((Ny - 1));
    }
  } else {
    Ny = 1;
    ys = 0;
    ny = 1;
  }

  std::vector<PetscInt> zvec;
  if(dim > 2) {
    if(zs == 0) {
      zvec.push_back(0);
    }
    if((zs + nz) == Nz) {
      zvec.push_back((Nz - 1));
    }
  } else {
    Nz = 1; 
    zs = 0;
    nz = 1;
  }

  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  //x
  for(int b = 0; b < xvec.size(); ++b) {
    int bXloc = xvec[b] - xs;
    for(int zi = zs; zi < (zs + nz); ++zi) {
      int bZloc = zi - zs;
      for(int yi = ys; yi < (ys + ny); ++yi) {
        int bYloc = yi - ys;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = ((offsets[rank] + bLoc)*dofsPerNode);
        for(int k = -1; k < 2; ++k) {
          int ok = zi + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yi + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi =  xvec[b] + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  int oBase = ((offsets[oPid] + oLoc)*dofsPerNode);
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
                      PetscInt oth = oBase + d;
                      if(k || j || i || d) {
                        MatSetValues(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      } else {
                        MatSetValues(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      }
                    }//end d
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end yi
    }//end zi
  }//end b

  //y
  for(int b = 0; b < yvec.size(); ++b) {
    int bYloc = yvec[b] - ys;
    for(int zi = zs; zi < (zs + nz); ++zi) {
      int bZloc = zi - zs;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = ((offsets[rank] + bLoc)*dofsPerNode);
        for(int k = -1; k < 2; ++k) {
          int ok = zi + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yvec[b] + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi = xi + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  int oBase = ((offsets[oPid] + oLoc)*dofsPerNode);
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
                      PetscInt oth = oBase + d;
                      if(k || j || i || d) {
                        MatSetValues(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      } else {
                        MatSetValues(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      }
                    }//end d
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end xi
    }//end zi
  }//end b

  //z
  for(int b = 0; b < zvec.size(); ++b) {
    int bZloc = zvec[b] - zs;
    for(int yi = ys; yi < (ys + ny); ++yi) {
      int bYloc = yi - ys;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = ((offsets[rank] + bLoc)*dofsPerNode);
        for(int k = -1; k < 2; ++k) {
          int ok = zvec[b] + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yi + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi = xi + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  int oBase = ((offsets[oPid] + oLoc)*dofsPerNode);
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
                      PetscInt oth = oBase + d;
                      if(k || j || i || d) {
                        MatSetValues(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      } else {
                        MatSetValues(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      }
                    }//end d
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end xi
    }//end yi
  }//end b

  MatAssemblyBegin(Kmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FINAL_ASSEMBLY);
}

void dirichletMatrixCorrectionBlkDiag(Mat Kblk, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<int>& offsets) {
  PetscInt dim;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  std::vector<PetscInt> xvec;
  if(xs == 0) {
    xvec.push_back(0);
  }
  if((xs + nx) == Nx) {
    xvec.push_back((Nx - 1));
  }

  std::vector<PetscInt> yvec;
  if(dim > 1) {
    if(ys == 0) {
      yvec.push_back(0);
    }
    if((ys + ny) == Ny) {
      yvec.push_back((Ny - 1));
    }
  } else {
    Ny = 1;
    ys = 0;
    ny = 1;
  }

  std::vector<PetscInt> zvec;
  if(dim > 2) {
    if(zs == 0) {
      zvec.push_back(0);
    }
    if((zs + nz) == Nz) {
      zvec.push_back((Nz - 1));
    }
  } else {
    Nz = 1; 
    zs = 0;
    nz = 1;
  }

  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  //x
  for(int b = 0; b < xvec.size(); ++b) {
    int bXloc = xvec[b] - xs;
    for(int zi = zs; zi < (zs + nz); ++zi) {
      int bZloc = zi - zs;
      for(int yi = ys; yi < (ys + ny); ++yi) {
        int bYloc = yi - ys;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        for(int k = -1; k < 2; ++k) {
          int ok = zi + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yi + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi =  xvec[b] + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  PetscInt oth = offsets[oPid] + oLoc;
                  if( (oi >= 0) && (oi < Nx) ) {
                    if(k || j || i) {
                      MatSetValues(Kblk, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                      MatSetValues(Kblk, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(Kblk, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                    }
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end yi
    }//end zi
  }//end b

  //y
  for(int b = 0; b < yvec.size(); ++b) {
    int bYloc = yvec[b] - ys;
    for(int zi = zs; zi < (zs + nz); ++zi) {
      int bZloc = zi - zs;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        for(int k = -1; k < 2; ++k) {
          int ok = zi + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yvec[b] + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi = xi + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  PetscInt oth = offsets[oPid] + oLoc;
                  if( (oi >= 0) && (oi < Nx) ) {
                    if(k || j || i) {
                      MatSetValues(Kblk, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                      MatSetValues(Kblk, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(Kblk, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                    }
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end xi
    }//end zi
  }//end b

  //z
  for(int b = 0; b < zvec.size(); ++b) {
    int bZloc = zvec[b] - zs;
    for(int yi = ys; yi < (ys + ny); ++yi) {
      int bYloc = yi - ys;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        for(int k = -1; k < 2; ++k) {
          int ok = zvec[b] + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yi + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi = xi + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  PetscInt oth = offsets[oPid] + oLoc;
                  if( (oi >= 0) && (oi < Nx) ) {
                    if(k || j || i) {
                      MatSetValues(Kblk, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                      MatSetValues(Kblk, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(Kblk, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                    }
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end xi
    }//end yi
  }//end b

  MatAssemblyBegin(Kblk, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Kblk, MAT_FINAL_ASSEMBLY);
}

void dirichletMatrixCorrectionBlkUpper(Mat Kblk, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<int>& offsets) {
  PetscInt dim;
  PetscInt dofsPerNode;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  std::vector<PetscInt> xvec;
  if(xs == 0) {
    xvec.push_back(0);
  }
  if((xs + nx) == Nx) {
    xvec.push_back((Nx - 1));
  }

  std::vector<PetscInt> yvec;
  if(dim > 1) {
    if(ys == 0) {
      yvec.push_back(0);
    }
    if((ys + ny) == Ny) {
      yvec.push_back((Ny - 1));
    }
  } else {
    Ny = 1;
    ys = 0;
    ny = 1;
  }

  std::vector<PetscInt> zvec;
  if(dim > 2) {
    if(zs == 0) {
      zvec.push_back(0);
    }
    if((zs + nz) == Nz) {
      zvec.push_back((Nz - 1));
    }
  } else {
    Nz = 1; 
    zs = 0;
    nz = 1;
  }

  PetscScalar zero = 0.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  //x
  for(int b = 0; b < xvec.size(); ++b) {
    int bXloc = xvec[b] - xs;
    for(int zi = zs; zi < (zs + nz); ++zi) {
      int bZloc = zi - zs;
      for(int yi = ys; yi < (ys + ny); ++yi) {
        int bYloc = yi - ys;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        for(int k = -1; k < 2; ++k) {
          int ok = zi + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yi + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi =  xvec[b] + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  int oBase = (offsets[oPid] + oLoc)*(dofsPerNode - 1);
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < (dofsPerNode - 1); ++d) {
                      PetscInt oth = oBase + d;
                      MatSetValues(Kblk, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    }//end d
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end yi
    }//end zi
  }//end b

  //y
  for(int b = 0; b < yvec.size(); ++b) {
    int bYloc = yvec[b] - ys;
    for(int zi = zs; zi < (zs + nz); ++zi) {
      int bZloc = zi - zs;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        for(int k = -1; k < 2; ++k) {
          int ok = zi + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yvec[b] + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi = xi + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  int oBase = (offsets[oPid] + oLoc)*(dofsPerNode - 1);
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < (dofsPerNode - 1); ++d) {
                      PetscInt oth = oBase + d;
                      MatSetValues(Kblk, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    }//end d
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end xi
    }//end zi
  }//end b

  //z
  for(int b = 0; b < zvec.size(); ++b) {
    int bZloc = zvec[b] - zs;
    for(int yi = ys; yi < (ys + ny); ++yi) {
      int bYloc = yi - ys;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        for(int k = -1; k < 2; ++k) {
          int ok = zvec[b] + k;
          int pk = rk;
          int oZs = zs;
          if(ok < zs) {
            --pk;
            oZs -= lz[pk];
          } else if(ok >= (zs + nz)) {
            ++pk;
            oZs += nz;
          }
          int oZloc = ok - oZs;
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              int oj = yi + j;
              int pj = rj;
              int oYs = ys;
              if(oj < ys) {
                --pj;
                oYs -= ly[pj];
              } else if(oj >= (ys + ny)) {
                ++pj;
                oYs += ny;
              }
              int oYloc = oj - oYs;
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  int oi = xi + i;
                  int pi = ri;
                  int oXs = xs;
                  if(oi < xs) {
                    --pi;
                    oXs -= lx[pi];
                  } else if(oi >= (xs + nx)) {
                    ++pi;
                    oXs += nx;
                  }
                  int oXloc = oi - oXs;
                  int oPid = (((pk*py) + pj)*px) + pi;
                  int oLoc = (((oZloc*ly[pj]) + oYloc)*lx[pi]) + oXloc;
                  int oBase = (offsets[oPid] + oLoc)*(dofsPerNode - 1);
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < (dofsPerNode - 1); ++d) {
                      PetscInt oth = oBase + d;
                      MatSetValues(Kblk, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    }//end d
                  }
                }//end i
              }
            }//end j
          }
        }//end k
      }//end xi
    }//end yi
  }//end b

  MatAssemblyBegin(Kblk, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Kblk, MAT_FINAL_ASSEMBLY);
}

void computeRandomRHS(DA da, Mat Kmat, Vec rhs, const unsigned int seed) {
  PetscRandom rndCtx;
  PetscRandomCreate(MPI_COMM_WORLD, &rndCtx);
  PetscRandomSetType(rndCtx, PETSCRAND48);
  PetscRandomSetSeed(rndCtx, seed);
  PetscRandomSeed(rndCtx);
  Vec tmpSol;
  VecDuplicate(rhs, &tmpSol);
  VecSetRandom(tmpSol, rndCtx);
  //VecSet(tmpSol, 10.0);
  PetscRandomDestroy(rndCtx);
  zeroBoundaries(da, tmpSol);
#ifdef DEBUG
  assert(Kmat != NULL);
#endif
  MatMult(Kmat, tmpSol, rhs);
  VecDestroy(tmpSol);
}

void zeroBoundaries(DA da, Vec vec) {
  PetscInt dim;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  if(dim == 1) {
    PetscScalar** arr; 
    DAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      arr[0][0] = 0.0;
    }
    if((xs + nx) == Nx) {
      arr[Nx - 1][0] = 0.0;
    }
    DAVecRestoreArrayDOF(da, vec, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr; 
    DAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        arr[yi][0][0] = 0.0;
      }//end yi
    }
    if((xs + nx) == Nx) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        arr[yi][Nx - 1][0] = 0.0;
      }//end yi
    }
    if(ys == 0) {
      for(int xi = xs; xi < (xs + nx); ++xi) {
        arr[0][xi][0] = 0.0;
      }//end xi
    }
    if((ys + ny) == Ny) {
      for(int xi = xs; xi < (xs + nx); ++xi) {
        arr[Ny - 1][xi][0] = 0.0;
      }//end xi
    }
    DAVecRestoreArrayDOF(da, vec, &arr);
  } else {
    PetscScalar**** arr; 
    DAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          arr[zi][yi][0][0] = 0.0;
        }//end yi
      }//end zi
    }
    if((xs + nx) == Nx) {
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int yi = ys; yi < (ys + ny); ++yi) {
          arr[zi][yi][Nx - 1][0] = 0.0;
        }//end yi
      }//end zi
    }
    if(ys == 0) {
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          arr[zi][0][xi][0] = 0.0;
        }//end xi
      }//end zi
    }
    if((ys + ny) == Ny) {
      for(int zi = zs; zi < (zs + nz); ++zi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          arr[zi][Ny - 1][xi][0] = 0.0;
        }//end xi
      }//end zi
    }
    if(zs == 0) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          arr[0][yi][xi][0] = 0.0;
        }//end xi
      }//end yi
    }
    if((zs + nz) == Nz) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int xi = xs; xi < (xs + nx); ++xi) {
          arr[Nz - 1][yi][xi][0] = 0.0;
        }//end xi
      }//end yi
    }
    DAVecRestoreArrayDOF(da, vec, &arr);
  }
}

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<KSP>& ksp, std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes) {
  PetscLogEventBegin(vCycleEvent, 0, 0, 0, 0);
#ifdef DEBUG
  assert(ksp[currLev] != NULL);
#endif
  KSPSolve(ksp[currLev], mgRhs[currLev], mgSol[currLev]);
  if(currLev > 0) {
    computeResidual(Kmat[currLev], mgSol[currLev], mgRhs[currLev], mgRes[currLev]);
    applyRestriction(Pmat[currLev - 1], tmpCvec[currLev - 1], mgRes[currLev], mgRhs[currLev - 1]);
    if(ksp[currLev - 1] != NULL) {
      if(currLev > 1) {
        VecZeroEntries(mgSol[currLev - 1]);
      }
      applyVcycle((currLev - 1), Kmat, Pmat, tmpCvec, ksp, mgSol, mgRhs, mgRes);
    }
    applyProlongation(Pmat[currLev - 1], tmpCvec[currLev - 1], mgSol[currLev - 1], mgRes[currLev]);
    VecAXPY(mgSol[currLev], 1.0, mgRes[currLev]);
    KSPSolve(ksp[currLev], mgRhs[currLev], mgSol[currLev]);
  }
  PetscLogEventEnd(vCycleEvent, 0, 0, 0, 0);
}

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res) {
  //res = rhs - (mat*sol)
  MatMult(mat, sol, res);
  VecAYPX(res, -1.0, rhs);
}

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms,
    std::vector<PCShellData>& data, int dim, int dofsPerNode, bool print) {
  PetscInt numSmoothIters = 2;
  PetscOptionsGetInt(PETSC_NULL, "-numSmoothIters", &numSmoothIters, PETSC_NULL);
  if(print) {
    std::cout<<"NumSmoothIters = "<<numSmoothIters<<std::endl;
  }
  ksp.resize((Kmat.size()), NULL);
  for(int lev = 0; lev < (Kmat.size()); ++lev) {
    if(Kmat[lev] != NULL) {
      PC pc;
      KSPCreate(activeComms[lev], &(ksp[lev]));
      KSPGetPC(ksp[lev], &pc);
      if(lev == 0) {
        KSPSetType(ksp[lev], KSPPREONLY);
        KSPSetInitialGuessNonzero(ksp[lev], PETSC_FALSE);
        PCSetType(pc, PCLU);
      } else {
        KSPSetType(ksp[lev], KSPFGMRES);
        KSPSetPreconditionerSide(ksp[lev], PC_RIGHT);
        PCSetType(pc, PCSHELL);
        PCShellSetContext(pc, &(data[lev - 1]));
        PCShellSetApply(pc, &applyShellPC);
        KSPSetInitialGuessNonzero(ksp[lev], PETSC_TRUE);
      }
      KSPSetOperators(ksp[lev], Kmat[lev], Kmat[lev], SAME_NONZERO_PATTERN);
      KSPSetTolerances(ksp[lev], 1.0e-12, 1.0e-12, PETSC_DEFAULT, numSmoothIters);
    }
  }//end lev
}

void createDA(std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dofsPerNode,
    int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<int> >& offsets,
    std::vector<std::vector<int> >& scanLz, std::vector<std::vector<int> >& scanLy, 
    std::vector<std::vector<int> >& scanLx, MPI_Comm globalComm, bool print) {
  PetscLogEventBegin(createDAevent, 0, 0, 0, 0);

  createGridSizes(dim, Nz, Ny, Nx, print);

  int globalRank;
  int globalNpes;
  MPI_Comm_rank(globalComm, &globalRank);
  MPI_Comm_size(globalComm, &globalNpes);

  int maxCoarseNpes = globalNpes;
  PetscOptionsGetInt(PETSC_NULL, "-maxCoarseNpes", &maxCoarseNpes, PETSC_NULL);
  if(maxCoarseNpes > globalNpes) {
    maxCoarseNpes = globalNpes;
  }
#ifdef DEBUG
  assert(maxCoarseNpes > 0);
#endif

  int numLevels = Nx.size();
#ifdef DEBUG
  assert(numLevels > 0);
#endif
  activeNpes.resize(numLevels);
  activeComms.resize(numLevels);
  da.resize(numLevels);
  partZ.resize(numLevels);
  partY.resize(numLevels);
  partX.resize(numLevels);
  offsets.resize(numLevels);
  scanLz.resize(numLevels);
  scanLy.resize(numLevels);
  scanLx.resize(numLevels);

  MPI_Group globalGroup;
  MPI_Comm_group(globalComm, &globalGroup);

  int* rankList = new int[globalNpes];
  for(int i = 0; i < globalNpes; ++i) {
    rankList[i] = i;
  }//end for i

  //0 is the coarsest level.
  for(int lev = 0; lev < numLevels; ++lev) {
    int maxNpes;
    if(lev == 0) {
      maxNpes = maxCoarseNpes;
    } else {
      maxNpes = globalNpes;
    }
    computePartition(dim, Nz[lev], Ny[lev], Nx[lev], maxNpes, partZ[lev], partY[lev], partX[lev],
        offsets[lev], scanLz[lev], scanLy[lev], scanLx[lev]);
    PetscInt pz = (partZ[lev]).size();
    PetscInt py = (partY[lev]).size();
    PetscInt px = (partX[lev]).size();
    activeNpes[lev] = (px*py*pz);
    if(print) {
      std::cout<<"Active Npes for Level "<<lev<<" = "<<(activeNpes[lev])
        <<" : (px, py, pz) = ("<<px<<", "<<py<<", "<<pz<<")"<<std::endl;
    }
#ifdef DEBUG
    if(lev > 0) {
      assert(activeNpes[lev] >= activeNpes[lev - 1]);
    }
#endif
    if(globalRank < (activeNpes[lev])) {
      MPI_Group subGroup;
      MPI_Group_incl(globalGroup, (activeNpes[lev]), rankList, &subGroup);
      MPI_Comm_create(globalComm, subGroup, &(activeComms[lev]));
      MPI_Group_free(&subGroup);
      DACreate(activeComms[lev], dim, DA_NONPERIODIC, DA_STENCIL_BOX, (Nx[lev]), (Ny[lev]), (Nz[lev]),
          px, py, pz, dofsPerNode, 1, &(partX[lev][0]), &(partY[lev][0]), &(partZ[lev][0]), (&(da[lev])));
    } else {
      MPI_Comm_create(globalComm, MPI_GROUP_EMPTY, &(activeComms[lev]));
#ifdef DEBUG
      assert(activeComms[lev] == MPI_COMM_NULL);
#endif
      da[lev] = NULL;
    }
  }//end lev

  delete [] rankList;
  MPI_Group_free(&globalGroup);

  PetscLogEventEnd(createDAevent, 0, 0, 0, 0);
}

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt> &lz, std::vector<PetscInt> &ly, std::vector<PetscInt> &lx,
    std::vector<int>& offsets, std::vector<int>& scanLz, std::vector<int>& scanLy, std::vector<int>& scanLx) {
#ifdef DEBUG
  if(dim < 3) {
    assert(Nz == 1);
  }
  if(dim < 2) {
    assert(Ny == 1);
  }
  assert(Nx > 0);
  assert(Ny > 0);
  assert(Nz > 0);
  assert(maxNpes > 0);
#endif

  std::vector<PetscInt> Nlist;
  Nlist.push_back(Nx);
  Nlist.push_back(Ny);
  Nlist.push_back(Nz);

  std::sort(Nlist.begin(), Nlist.end());

  double tmp = std::pow(((static_cast<double>(Nx*Ny*Nz))/(static_cast<double>(maxNpes))), (1.0/(static_cast<double>(dim))));

  std::vector<int> pList(3, 1);
  for(int d = 0; d < 3; ++d) {
    if(Nlist[d] > 1) {
      pList[d] = static_cast<int>(std::floor((static_cast<double>(Nlist[d]))/tmp));
      if(pList[d] > Nlist[d]) {
        pList[d] = Nlist[d];
      }
      if(pList[d] < 1) {
        pList[d] = 1;
      }
    }
  }//end d
#ifdef DEBUG
  assert(((pList[0])*(pList[1])*(pList[2])) <= maxNpes);
#endif

  bool partChanged;
  do {
    partChanged = false;
    for(int d = 2; d >= 0; --d) {
      if( pList[d] < Nlist[d] ) {
        if( ((pList[d] + 1)*(pList[(d+1)%3])*(pList[(d+2)%3])) <= maxNpes ) {
          ++(pList[d]);
          partChanged = true;
        }
      }
    }//end d
  } while(partChanged);

  int px;
  for(int d = 0; d < 3; ++d) {
    if(Nx == Nlist[d]) {
      px = pList[d];
      Nlist.erase(Nlist.begin() + d);
      pList.erase(pList.begin() + d);
      break;
    }
  }//end d

  int py;
  for(int d = 0; d < 2; ++d) {
    if(Ny == Nlist[d]) {
      py = pList[d];
      Nlist.erase(Nlist.begin() + d);
      pList.erase(pList.begin() + d);
      break;
    }
  }//end d

#ifdef DEBUG
  assert(Nz == Nlist[0]);
#endif

  int pz;
  pz = pList[0];

#ifdef DEBUG
  assert((px*py*pz) <= maxNpes);
  assert(px >= 1);
  assert(py >= 1);
  assert(pz >= 1);
  assert(px <= Nx);
  assert(py <= Ny);
  assert(pz <= Nz);
#endif

  PetscInt avgX = Nx/px;
  PetscInt extraX = Nx%px; 
  lx.resize(px, avgX);
  for(int cnt = 0; cnt < extraX; ++cnt) {
    ++(lx[cnt]);
  }//end cnt

  PetscInt avgY = Ny/py;
  PetscInt extraY = Ny%py; 
  ly.resize(py, avgY);
  for(int cnt = 0; cnt < extraY; ++cnt) {
    ++(ly[cnt]);
  }//end cnt

  PetscInt avgZ = Nz/pz;
  PetscInt extraZ = Nz%pz; 
  lz.resize(pz, avgZ);
  for(int cnt = 0; cnt < extraZ; ++cnt) {
    ++(lz[cnt]);
  }//end cnt

  int npes = px*py*pz;

  offsets.resize(npes);
  offsets[0] = 0;
  for(int p = 1; p < npes; ++p) {
    int k = (p - 1)/(px*py);
    int j = ((p - 1)/px)%py;
    int i = (p - 1)%px;
    offsets[p] = offsets[p - 1] + (lz[k]*ly[j]*lx[i]);
  }//end p

  scanLx.resize(px);
  scanLx[0] = lx[0] - 1;
  for(int i = 1; i < px; ++i) {
    scanLx[i] = scanLx[i - 1] + lx[i];
  }//end i

  scanLy.resize(py);
  scanLy[0] = ly[0] - 1;
  for(int i = 1; i < py; ++i) {
    scanLy[i] = scanLy[i - 1] + ly[i];
  }//end i

  scanLz.resize(pz);
  scanLz[0] = lz[0] - 1;
  for(int i = 1; i < pz; ++i) {
    scanLz[i] = scanLz[i - 1] + lz[i];
  }//end i
}

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, bool print) {
#ifdef DEBUG
  assert(dim > 0);
  assert(dim <= 3);
#endif

  PetscInt currNx = 17;
  PetscInt currNy = 1;
  PetscInt currNz = 1;

  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &currNx, PETSC_NULL);
  if(print) {
    std::cout<<"Nx (Finest) = "<<currNx<<std::endl;
  }
  if(dim > 1) {
    PetscOptionsGetInt(PETSC_NULL, "-finestNy", &currNy, PETSC_NULL);
    if(print) {
      std::cout<<"Ny (Finest) = "<<currNy<<std::endl;
    }
  }
  if(dim > 2) {
    PetscOptionsGetInt(PETSC_NULL, "-finestNz", &currNz, PETSC_NULL);
    if(print) {
      std::cout<<"Nz (Finest) = "<<currNz<<std::endl;
    }
  }

  PetscInt maxNumLevels = 20;
  PetscOptionsGetInt(PETSC_NULL, "-maxNumLevels", &maxNumLevels, PETSC_NULL);
  if(print) {
    std::cout<<"MaxNumLevels = "<<maxNumLevels<<std::endl;
  }

  const unsigned int minGridSize = 9;

  Nx.clear();
  Ny.clear();
  Nz.clear();

  //0 is the coarsest level.
  for(int lev = 0; lev < maxNumLevels; ++lev) {
    Nx.insert(Nx.begin(), currNx);
    if(dim > 1) {
      Ny.insert(Ny.begin(), currNy);
    }
    if(dim > 2) {
      Nz.insert(Nz.begin(), currNz);
    }
    if( (currNx < minGridSize) || ((currNx%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
    if(dim > 1) {
      if( (currNy < minGridSize) || ((currNy%2) == 0) ) {
        break;
      }
      currNy = 1 + ((currNy - 1)/2); 
    }
    if(dim > 2) {
      if( (currNz < minGridSize) || ((currNz%2) == 0) ) {
        break;
      }
      currNz = 1 + ((currNz - 1)/2); 
    }
  }//lev

#ifdef DEBUG
  if(dim < 2) {
    assert(Ny.empty());
  } else { 
    assert( (Ny.size()) == (Nx.size()) );
  }
  if(dim < 3) {
    assert(Nz.empty());
  } else {
    assert( (Nz.size()) == (Nx.size()) );
  }
#endif

  if(dim < 2) {
    Ny.resize((Nx.size()), 1);
  }
  if(dim < 3) {
    Nz.resize((Nx.size()), 1);
  }

  if(print) {
    std::cout<<"ActualNumLevels = "<<(Nx.size())<<std::endl;
  }
}

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes) {
  mgSol.resize(Kmat.size(), NULL);
  mgRhs.resize(Kmat.size(), NULL);
  mgRes.resize(Kmat.size(), NULL);
  for(int i = 0; i < (Kmat.size() - 1); ++i) {
    if(Kmat[i] != NULL) {
      MatGetVecs(Kmat[i], &(mgSol[i]), &(mgRhs[i]));
      VecDuplicate(mgRhs[i], &(mgRes[i]));
    }
  }//end i
}

void destroyComms(std::vector<MPI_Comm> & activeComms) {
  for(int i = 0; i < activeComms.size(); ++i) {
    if(activeComms[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&(activeComms[i]));
    }
  }//end i
  activeComms.clear();
}

void destroyVec(std::vector<Vec>& vec) {
  for(int i = 0; i < vec.size(); ++i) {
    if(vec[i] != NULL) {
      VecDestroy(vec[i]);
    }
  }//end i
  vec.clear();
}

void destroyMat(std::vector<Mat> & mat) {
  for(int i = 0; i < mat.size(); ++i) {
    if(mat[i] != NULL) {
      MatDestroy(mat[i]);
    }
  }//end i
  mat.clear();
}

void destroyDA(std::vector<DA>& da) {
  for(int i = 0; i < da.size(); ++i) {
    if(da[i] != NULL) {
      DADestroy(da[i]);
    }
  }//end i
  da.clear();
}

void destroyKSP(std::vector<KSP>& ksp) {
  for(int i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(ksp[i]);
    }
  }//end i
  ksp.clear();
}

void destroyPCShellData(std::vector<PCShellData>& data) {
}

/*
   void mySolver(Mat A, Vec rhs, Vec sol) {
   Vec diag;
   Vec res;
   VecDuplicate(sol, &diag);
   VecDuplicate(sol, &res);
   MatGetDiagonal(A, diag);
   PetscInt vecSz;
   VecGetSize(sol, &vecSz);

   PetscInt K;
   PetscOptionsGetInt(PETSC_NULL, "-K", &K, PETSC_NULL);

   PetscInt numSmooth = 2;
   PetscOptionsGetInt(PETSC_NULL, "-numSmooth", &numSmooth, PETSC_NULL);

   PetscScalar alpha;

   std::vector<double> omega(K + 1);
   if(K == 0) {
   omega[0] = (2.0/3.0);
   } else if(K == 1) {
   omega[0] = (2.0/3.0);
   omega[1] = 1.0;
   } else if(K == 2) {
   omega[0] = 0.8;
   omega[1] = 1.0;
   omega[2] = 1.0;
   } else if(K == 3) {
   PetscOptionsGetScalar(PETSC_NULL, "-alpha0", &alpha, PETSC_NULL);
   omega[0] = alpha;
   PetscOptionsGetScalar(PETSC_NULL, "-alpha1", &alpha, PETSC_NULL);
   omega[1] = alpha;
   PetscOptionsGetScalar(PETSC_NULL, "-alpha2", &alpha, PETSC_NULL);
   omega[2] = alpha;
   PetscOptionsGetScalar(PETSC_NULL, "-alpha3", &alpha, PETSC_NULL);
   omega[3] = alpha;
   }

   PetscScalar* diagArr;
   VecGetArray(diag, &diagArr);

   PetscScalar* solArr;
   PetscScalar* resArr;
   for(int iter = 0; iter < numSmooth; ++iter) {
   for(int d = 0; d <= K; ++d) {
   computeResidual(A, sol, rhs, res);
   VecGetArray(sol, &solArr);
   VecGetArray(res, &resArr);
   for(int i = 0; i < vecSz; ++i) {
   if((i%(K + 1)) == d) {
   solArr[i] += (omega[d]*resArr[i]/diagArr[i]);
   }
   }//end i
   VecRestoreArray(sol, &solArr);
   VecRestoreArray(res, &resArr);
   }//end d
   }//end iter

   VecRestoreArray(diag, &diagArr);

   VecDestroy(diag);
   VecDestroy(res);
   }
   */

PetscErrorCode applyShellPC(void* ctx, Vec in, Vec out) {
  return 0;
}


