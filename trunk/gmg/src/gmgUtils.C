
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"
#include "petscmg.h"

#ifdef DEBUG
#include <cassert>
#endif

extern PetscLogEvent createDAevent;
extern PetscLogEvent buildPmatEvent;
extern PetscLogEvent PmemEvent;
extern PetscLogEvent fillPmatEvent;
extern PetscLogEvent buildKmatEvent;
extern PetscLogEvent KmemEvent;
extern PetscLogEvent fillKmatEvent;
extern PetscLogEvent elemKmatEvent;
extern PetscLogEvent dirichletMatCorrectionEvent;
extern PetscLogEvent vCycleEvent;

void buildKmat(std::vector<unsigned long long int>& factorialsList,
    std::vector<Mat>& Kmat, std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<std::vector<PetscInt> >& lz, std::vector<std::vector<PetscInt> >& ly, std::vector<std::vector<PetscInt> >& lx,
    std::vector<std::vector<int> >& offsets, bool print) {
  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  Kmat.resize(da.size(), NULL);
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      PetscLogEventBegin(KmemEvent, 0, 0, 0, 0);
#ifdef USE_STENCIL
      DAGetMatrix(da[i], MATAIJ, &(Kmat[i]));
#else
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
      int factor = 3;
      if(dim > 1) {
        factor *= 3;
      }
      if(dim > 2) {
        factor *= 3;
      }
      if(activeNpes[i] > 1) {
        MatMPIAIJSetPreallocation(Kmat[i], (factor*dofsPerNode), PETSC_NULL, (factor*dofsPerNode), PETSC_NULL);
      } else {
        MatSeqAIJSetPreallocation(Kmat[i], (factor*dofsPerNode), PETSC_NULL);
      }
#endif
      PetscLogEventEnd(KmemEvent, 0, 0, 0, 0);
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

void buildPmat(std::vector<unsigned long long int>& factorialsList, 
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, std::vector<std::vector<PetscInt> >& partZ,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<int> >& offsets,
    std::vector<std::vector<int> >& scanLz, std::vector<std::vector<int> >& scanLy, std::vector<std::vector<int> >& scanLx, bool print) {
  PetscLogEventBegin(buildPmatEvent, 0, 0, 0, 0);

  Pmat.resize((da.size() - 1), NULL);
  tmpCvec.resize(Pmat.size(), NULL);
  for(int lev = 0; lev < (Pmat.size()); ++lev) {
    if(da[lev + 1] != NULL) {
      PetscLogEventBegin(PmemEvent, 0, 0, 0, 0);
      PetscInt nxf, nyf, nzf;
      DAGetCorners(da[lev + 1], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nxf, &nyf, &nzf);
      MatCreate(activeComms[lev + 1], &(Pmat[lev]));
      PetscInt nxc, nyc, nzc;
      nxc = nyc = nzc = 0;
      if(da[lev] != NULL) {
        DAGetCorners(da[lev], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nxc, &nyc, &nzc);
      }
      if(dim < 3) {
        nzf = nzc = 1;
      }
      if(dim < 2) {
        nyf = nyc = 1;
      }
      PetscInt locRowSz = dofsPerNode*nxf*nyf*nzf;
      PetscInt locColSz = dofsPerNode*nxc*nyc*nzc;
      MatSetSizes(Pmat[lev], locRowSz, locColSz, PETSC_DETERMINE, PETSC_DETERMINE);
      MatSetType(Pmat[lev], MATAIJ);
      int dofsPerElem = (1 << dim);
      if(activeNpes[lev + 1] > 1) {
        MatMPIAIJSetPreallocation(Pmat[lev], (dofsPerElem*dofsPerNode), PETSC_NULL, (dofsPerElem*dofsPerNode), PETSC_NULL);
      } else {
        MatSeqAIJSetPreallocation(Pmat[lev], (dofsPerElem*dofsPerNode), PETSC_NULL);
      }
      MatGetVecs(Pmat[lev], &(tmpCvec[lev]), PETSC_NULL);
      PetscLogEventEnd(PmemEvent, 0, 0, 0, 0);
      computePmat(factorialsList, Pmat[lev], Nz[lev], Ny[lev], Nx[lev], Nz[lev + 1], Ny[lev + 1], Nx[lev + 1],
          partZ[lev], partY[lev], partX[lev], partZ[lev + 1], partY[lev + 1], partX[lev + 1],
          offsets[lev], scanLz[lev], scanLy[lev], scanLx[lev],
          offsets[lev + 1], scanLz[lev + 1], scanLy[lev + 1], scanLx[lev + 1],
          dim, dofsPerNode, coeffs, K);
    }
    if(print) {
      std::cout<<"Built Pmat for level = "<<lev<<std::endl;
    }
  }//end lev

  PetscLogEventEnd(buildPmatEvent, 0, 0, 0, 0);
}

void computePmat(std::vector<unsigned long long int>& factorialsList, 
    Mat Pmat, int Nzc, int Nyc, int Nxc, int Nzf, int Nyf, int Nxf,
    std::vector<PetscInt>& lzc, std::vector<PetscInt>& lyc, std::vector<PetscInt>& lxc,
    std::vector<PetscInt>& lzf, std::vector<PetscInt>& lyf, std::vector<PetscInt>& lxf,
    std::vector<int>& cOffsets, std::vector<int>& scanClz, std::vector<int>& scanCly, std::vector<int>& scanClx,
    std::vector<int>& fOffsets, std::vector<int>& scanFlz, std::vector<int>& scanFly, std::vector<int>& scanFlx,
    int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K) {
  PetscLogEventBegin(fillPmatEvent, 0, 0, 0, 0);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int fpx = lxf.size();
  int fpy = lyf.size();
  int fpz = lzf.size();

  int fpk = rank/(fpx*fpy);
  int fpj = (rank/fpx)%fpy;
  int fpi = rank%fpx;

  int fnz = lzf[fpk];
  int fny = lyf[fpj];
  int fnx = lxf[fpi];

  int fzs = 0;
  if(fpk > 0) {
    fzs = 1 + scanFlz[fpk - 1];
  }

  int fys = 0;
  if(fpj > 0) {
    fys = 1 + scanFly[fpj - 1];
  }

  int fxs = 0;
  if(fpi > 0) {
    fxs = 1 + scanFlx[fpi - 1];
  }

  int fOff = fOffsets[rank];

  int cpx = lxc.size();
  int cpy = lyc.size();
  int cpz = lzc.size();

  std::vector<long double> factorX(K + 1);
  std::vector<long double> factorY;
  std::vector<long double> factorZ;
  long double hxf, hyf, hzf;
  long double hxc, hyc, hzc;
  hxf = 1.0L/(static_cast<long double>(Nxf - 1));
  hxc = 1.0L/(static_cast<long double>(Nxc - 1));
  if(dim > 1) {
    hyf = 1.0L/(static_cast<long double>(Nyf - 1));
    hyc = 1.0L/(static_cast<long double>(Nyc - 1));
    factorY.resize(K + 1);
  } else {
    hyf = 1.0L;
    hyc = 1.0L;
    factorY.resize(1);
  }
  if(dim > 2) {
    hzf = 1.0L/(static_cast<long double>(Nzf - 1));
    hzc = 1.0L/(static_cast<long double>(Nzc - 1));
    factorZ.resize(K + 1);
  } else {
    hzf = 1.0L;
    hzc = 1.0L;
    factorZ.resize(1);
  }

  for(int i = 0; i < factorX.size(); ++i) {
    factorX[i] = myIntPow((hxf/hxc), i);
  }//end i
  for(int i = 0; i < factorY.size(); ++i) {
    factorY[i] = myIntPow((hyf/hyc), i);
  }//end i
  for(int i = 0; i < factorZ.size(); ++i) {
    factorZ[i] = myIntPow((hzf/hzc), i);
  }//end i

  long double pt[] = {-1.0, 0.0, 1.0};

  std::vector<std::vector<std::vector<std::vector<long double> > > > eval1Dderivatives(2);
  for(int nodeId = 0; nodeId < 2; ++nodeId) {
    eval1Dderivatives[nodeId].resize(K + 1);
    for(int cdof = 0; cdof <= K; ++cdof) {
      eval1Dderivatives[nodeId][cdof].resize(3);
      for(int ptId = 0; ptId < 3; ++ptId) {
        eval1Dderivatives[nodeId][cdof][ptId].resize(K + 1);
        for(int fdof = 0; fdof <= K; ++fdof) {
          eval1Dderivatives[nodeId][cdof][ptId][fdof] = eval1DshFnLderivative(factorialsList, 
              nodeId, cdof, K, coeffs, pt[ptId], fdof);
        }//end fdof
      }//end pt
    }//end cdof
  }//end nodeId

  MatZeroEntries(Pmat);

  for(int fzi = fzs; fzi < (fzs + fnz); ++fzi) {
    int czi = fzi/2;
    bool oddZ = ((fzi%2) != 0);
    std::vector<int>::iterator zIt = std::lower_bound(scanClz.begin(), scanClz.end(), czi);
#ifdef DEBUG
    assert(zIt != scanClz.end());
#endif
    std::vector<int> zVec;
    std::vector<int> zPid;
    zVec.push_back(czi);
    zPid.push_back((zIt - scanClz.begin()));
    if(oddZ) {
      zVec.push_back(czi + 1);
      if((*zIt) == czi) {
        zPid.push_back((zIt - scanClz.begin() + 1));
      } else {
        zPid.push_back((zIt - scanClz.begin()));
      }
    }
    for(int fyi = fys; fyi < (fys + fny); ++fyi) {
      int cyi = fyi/2;
      bool oddY = ((fyi%2) != 0);
      std::vector<int>::iterator yIt = std::lower_bound(scanCly.begin(), scanCly.end(), cyi);
#ifdef DEBUG
      assert(yIt != scanCly.end());
#endif
      std::vector<int> yVec;
      std::vector<int> yPid;
      yVec.push_back(cyi);
      yPid.push_back((yIt - scanCly.begin()));
      if(oddY) {
        yVec.push_back(cyi + 1);
        if((*yIt) == cyi) {
          yPid.push_back((yIt - scanCly.begin() + 1));
        } else {
          yPid.push_back((yIt - scanCly.begin()));
        }
      }
      for(int fxi = fxs; fxi < (fxs + fnx); ++fxi) {
        int cxi = fxi/2;
        bool oddX = ((fxi%2) != 0);
        std::vector<int>::iterator xIt = std::lower_bound(scanClx.begin(), scanClx.end(), cxi);
#ifdef DEBUG
        assert(xIt != scanClx.end());
#endif
        std::vector<int> xVec;
        std::vector<int> xPid;
        xVec.push_back(cxi);
        xPid.push_back((xIt - scanClx.begin()));
        if(oddX) {
          xVec.push_back(cxi + 1);
          if((*xIt) == cxi) {
            xPid.push_back((xIt - scanClx.begin() + 1));
          } else {
            xPid.push_back((xIt - scanClx.begin()));
          }
        }
        int fLoc = ((((fzi - fzs)*fny) + (fyi - fys))*fnx) + (fxi - fxs);
        for(int fd = 0; fd < dofsPerNode; ++fd) {
          bool isFineBoundary = false;
          if(fd == 0) {
            if( (fxi == 0) || (fxi == (Nxf - 1)) ) {
              isFineBoundary = true;
            }
            if( (dim > 1) && ( (fyi == 0) || (fyi == (Nyf - 1)) ) ) {
              isFineBoundary = true;
            }
            if( (dim > 2) && ( (fzi == 0) || (fzi == (Nzf - 1)) ) ) {
              isFineBoundary = true;
            }
          }
          if(isFineBoundary) {
            continue;
          }
          int zfd = fd/((K + 1)*(K + 1));
          int yfd = (fd/(K + 1))%(K + 1);
          int xfd = fd%(K + 1);
          int rowId = ((fOff + fLoc)*dofsPerNode) + fd;
          for(int k = 0; k < zVec.size(); ++k) {
            int zLoc;
            if(zPid[k] > 0) {
              zLoc = zVec[k] - (1 + scanClz[zPid[k] - 1]);
            } else {
              zLoc = zVec[k];
            }
            for(int j = 0; j < yVec.size(); ++j) {
              int yLoc;
              if(yPid[j] > 0) {
                yLoc = yVec[j] - (1 + scanCly[yPid[j] - 1]);
              } else {
                yLoc = yVec[j];
              }
              for(int i = 0; i < xVec.size(); ++i) {
                int xLoc;
                if(xPid[i] > 0) {
                  xLoc = xVec[i] - (1 + scanClx[xPid[i] - 1]);
                } else {
                  xLoc = xVec[i];
                }
                int cPid = (((zPid[k]*cpy) + yPid[j])*cpx) + xPid[i];
                int cLoc = (((zLoc*lyc[yPid[j]]) + yLoc)*lxc[xPid[i]]) + xLoc;
                for(int d = 0; d < dofsPerNode; ++d) {
                  bool isCoarseBoundary = false;
                  if(d == 0) {
                    if( (xVec[i] == 0) || (xVec[i] == (Nxc - 1)) ) {
                      isCoarseBoundary = true;
                    }
                    if( (dim > 1) && ( (yVec[j] == 0) || (yVec[j] == (Nyc - 1)) ) ) {
                      isCoarseBoundary = true;
                    }
                    if( (dim > 2) && ( (zVec[k] == 0) || (zVec[k] == (Nzc - 1)) ) ) {
                      isCoarseBoundary = true;
                    }
                  }
                  if(isCoarseBoundary) {
                    continue;
                  }
                  int zcd = d/((K + 1)*(K + 1));
                  int ycd = (d/(K + 1))%(K + 1);
                  int xcd = d%(K + 1);
                  int colId = ((cOffsets[cPid] + cLoc)*dofsPerNode) + d;
                  int xNodeId;
                  if( (xVec[i] == (Nxc - 1)) || i ) {
                    xNodeId = 1;
                  } else {
                    xNodeId = 0;
                  }
                  int yNodeId;
                  if( (yVec[i] == (Nyc - 1)) || j ) {
                    yNodeId = 1;
                  } else {
                    yNodeId = 0;
                  }
                  int zNodeId;
                  if( (zVec[i] == (Nzc - 1)) || k ) {
                    zNodeId = 1;
                  } else {
                    zNodeId = 0;
                  }
                  int xPtId;
                  if(oddX) {
                    xPtId = 1;
                  } else {
                    if(xNodeId == 0) {
                      xPtId = 0;
                    } else {
                      xPtId = 2;
                    }
                  }
                  int yPtId;
                  if(oddY) {
                    yPtId = 1;
                  } else {
                    if(yNodeId == 0) {
                      yPtId = 0;
                    } else {
                      yPtId = 2;
                    }
                  }
                  int zPtId;
                  if(oddZ) {
                    zPtId = 1;
                  } else {
                    if(zNodeId == 0) {
                      zPtId = 0;
                    } else {
                      zPtId = 2;
                    }
                  }
                  long double val = (factorX[xfd] * eval1Dderivatives[xNodeId][xcd][xPtId][xfd]);
                  if(dim > 1) {
                    val *= (factorY[yfd] * eval1Dderivatives[yNodeId][ycd][yPtId][yfd]);
                  } 
                  if(dim > 2) {
                    val *= (factorZ[zfd] * eval1Dderivatives[zNodeId][zcd][zPtId][zfd]);
                  }
                  PetscScalar val2 = val;
                  MatSetValues(Pmat, 1, &rowId, 1, &colId, &val2, INSERT_VALUES);
                }//end d
              }//end i
            }//end j
          }//end k
        }//end fd
      }//end fxi
    }//end fyi
  }//end fzi

  MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY);

  PetscLogEventEnd(fillPmatEvent, 0, 0, 0, 0);
}

void computeKmat(std::vector<unsigned long long int>& factorialsList,
    Mat Kmat, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<int>& offsets, std::vector<long long int>& coeffs, const unsigned int K, bool print) {
  PetscLogEventBegin(fillKmatEvent, 0, 0, 0, 0);

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

  PetscLogEventBegin(elemKmatEvent, 0, 0, 0, 0);
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
  PetscLogEventEnd(elemKmatEvent, 0, 0, 0, 0);

  unsigned int nodesPerElem = (1 << dim);

#ifdef USE_STENCIL
  std::vector<MatStencil> indices(nodesPerElem*dofsPerNode);
#else
  std::vector<PetscInt> indices(nodesPerElem*dofsPerNode);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;
#endif

  MatZeroEntries(Kmat);

  for(unsigned int zi = zs; zi < (zs + nze); ++zi) {
    for(unsigned int yi = ys; yi < (ys + nye); ++yi) {
      for(unsigned int xi = xs; xi < (xs + nxe); ++xi) {
        for(int z = 0, i = 0; z < numZnodes; ++z) {
          int vk = (zi + z);
#ifndef USE_STENCIL
          int pk = rk;
          int vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          int zLoc = vk - vZs;
#endif
          for(int y = 0; y < numYnodes; ++y) {
            int vj = (yi + y);
#ifndef USE_STENCIL
            int pj = rj;
            int vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            int yLoc = vj - vYs;
#endif
            for(int x = 0; x < numXnodes; ++x) {
              int vi = (xi + x);
#ifndef USE_STENCIL
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
#endif
              for(unsigned int d = 0; d < dofsPerNode; ++i, ++d) {
#ifdef USE_STENCIL
                (indices[i]).k = vk;
                (indices[i]).j = vj;
                (indices[i]).i = vi;
                (indices[i]).c = d;
#else
                indices[i] = idBase + d;
#endif
              }//end d
            }//end x
          }//end y
        }//end z
#ifdef USE_STENCIL
        MatSetValuesStencil(Kmat, (indices.size()), &(indices[0]),
            (indices.size()), &(indices[0]), &(vals[0]), ADD_VALUES);
#else
        MatSetValues(Kmat, (indices.size()), &(indices[0]),
            (indices.size()), &(indices[0]), &(vals[0]), ADD_VALUES);
#endif
      }//end xi
    }//end yi
  }//end zi

  MatAssemblyBegin(Kmat, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FLUSH_ASSEMBLY);

  PetscLogEventEnd(fillKmatEvent, 0, 0, 0, 0);
}

void dirichletMatrixCorrection(Mat Kmat, DA da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<int>& offsets) {
  PetscLogEventBegin(dirichletMatCorrectionEvent, 0, 0, 0, 0);

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

#ifdef USE_STENCIL
  MatStencil oth;
  MatStencil bnd;
  bnd.c = 0;
#else
  PetscInt oth;
  PetscInt bnd;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();
  int pz = lz.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;
#endif

  //x
  for(int b = 0; b < xvec.size(); ++b) {
#ifdef USE_STENCIL
    bnd.i = xvec[b];
#else
    int bXloc = xvec[b] - xs;
#endif
    for(int zi = zs; zi < (zs + nz); ++zi) {
#ifdef USE_STENCIL
      bnd.k = zi;
#else
      int bZloc = zi - zs;
#endif
      for(int yi = ys; yi < (ys + ny); ++yi) {
#ifdef USE_STENCIL
        bnd.j = yi;
#else
        int bYloc = yi - ys;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        bnd = ((offsets[rank] + bLoc)*dofsPerNode);
#endif
        for(int k = -1; k < 2; ++k) {
#ifdef USE_STENCIL
          int ok = (bnd.k) + k;
          oth.k = ok;
#else
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
#endif
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
#ifdef USE_STENCIL
              int oj = (bnd.j) + j;
              oth.j = oj;
#else
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
#endif
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
#ifdef USE_STENCIL
                  int oi = (bnd.i) + i; 
                  oth.i = oi;
#else
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
#endif
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
#ifdef USE_STENCIL
                      oth.c = d;
#else
                      oth = oBase + d;
#endif
                      if(k || j || i || d) {
#ifdef USE_STENCIL
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
#else
                        MatSetValues(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
#endif
                      } else {
#ifdef USE_STENCIL
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
#else
                        MatSetValues(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
#endif
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
#ifdef USE_STENCIL
    bnd.j = yvec[b];
#else
    int bYloc = yvec[b] - ys;
#endif
    for(int zi = zs; zi < (zs + nz); ++zi) {
#ifdef USE_STENCIL
      bnd.k = zi;
#else
      int bZloc = zi - zs;
#endif
      for(int xi = xs; xi < (xs + nx); ++xi) {
#ifdef USE_STENCIL
        bnd.i = xi; 
#else
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        bnd = ((offsets[rank] + bLoc)*dofsPerNode);
#endif
        for(int k = -1; k < 2; ++k) {
#ifdef USE_STENCIL
          int ok = (bnd.k) + k;
          oth.k = ok;
#else
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
#endif
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
#ifdef USE_STENCIL
              int oj = (bnd.j) + j;
              oth.j = oj;
#else
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
#endif
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
#ifdef USE_STENCIL
                  int oi = (bnd.i) + i;
                  oth.i = oi;
#else
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
#endif
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
#ifdef USE_STENCIL
                      oth.c = d;
#else
                      oth = oBase + d;
#endif
                      if(k || j || i || d) {
#ifdef USE_STENCIL
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
#else
                        MatSetValues(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
#endif
                      } else {
#ifdef USE_STENCIL
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
#else
                        MatSetValues(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
#endif
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
#ifdef USE_STENCIL
    bnd.k = zvec[b];
#else
    int bZloc = zvec[b] - zs;
#endif
    for(int yi = ys; yi < (ys + ny); ++yi) {
#ifdef USE_STENCIL
      bnd.j = yi;
#else
      int bYloc = yi - ys;
#endif
      for(int xi = xs; xi < (xs + nx); ++xi) {
#ifdef USE_STENCIL
        bnd.i = xi; 
#else
        int bXloc = xi - xs;
        int bLoc = (((bZloc*ny) + bYloc)*nx) + bXloc;
        bnd = ((offsets[rank] + bLoc)*dofsPerNode);
#endif
        for(int k = -1; k < 2; ++k) {
#ifdef USE_STENCIL
          int ok = (bnd.k) + k;
          oth.k = ok;
#else
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
#endif
          if( (ok >= 0) && (ok < Nz) ) {
            for(int j = -1; j < 2; ++j) {
#ifdef USE_STENCIL
              int oj = (bnd.j) + j; 
              oth.j = oj;
#else
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
#endif
              if( (oj >= 0) && (oj < Ny) ) {
                for(int i = -1; i < 2; ++i) {
#ifdef USE_STENCIL
                  int oi = (bnd.i) + i;
                  oth.i = oi;
#else
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
#endif
                  if( (oi >= 0) && (oi < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
#ifdef USE_STENCIL
                      oth.c = d;
#else
                      oth = oBase + d;
#endif
                      if(k || j || i || d) {
#ifdef USE_STENCIL
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
#else
                        MatSetValues(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
#endif
                      } else {
#ifdef USE_STENCIL
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
#else
                        MatSetValues(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
#endif
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

  PetscLogEventEnd(dirichletMatCorrectionEvent, 0, 0, 0, 0);
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
      VecZeroEntries(mgSol[currLev - 1]);
      applyVcycle((currLev - 1), Kmat, Pmat, tmpCvec, ksp, mgSol, mgRhs, mgRes);
    }
    applyProlongation(Pmat[currLev - 1], tmpCvec[currLev - 1], mgSol[currLev - 1], mgRes[currLev]);
    VecAXPY(mgSol[currLev], 1.0, mgRes[currLev]);
    KSPSolve(ksp[currLev], mgRhs[currLev], mgSol[currLev]);
  }
  PetscLogEventEnd(vCycleEvent, 0, 0, 0, 0);
}

void applyRestriction(Mat Pmat, Vec tmpCvec, Vec fVec, Vec cVec) {
#ifdef DEBUG
  assert(Pmat != NULL);
  assert(fVec != NULL);
  assert(tmpCvec != NULL);
#endif
  PetscScalar* arr;
  if(cVec != NULL) {
    VecGetArray(cVec, &arr);
    VecPlaceArray(tmpCvec, arr);
  }
  MatMultTranspose(Pmat, fVec, tmpCvec);
  if(cVec != NULL) {
    VecResetArray(tmpCvec);
    VecRestoreArray(cVec, &arr);
  }
}

void applyProlongation(Mat Pmat, Vec tmpCvec, Vec cVec, Vec fVec) {
#ifdef DEBUG
  assert(Pmat != NULL);
  assert(fVec != NULL);
  assert(tmpCvec != NULL);
#endif
  PetscScalar* arr;
  if(cVec != NULL) {
    VecGetArray(cVec, &arr);
    VecPlaceArray(tmpCvec, arr);
  }
  MatMult(Pmat, tmpCvec, fVec);
  if(cVec != NULL) {
    VecResetArray(tmpCvec);
    VecRestoreArray(cVec, &arr);
  }
}

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res) {
  //res = rhs - (mat*sol)
  MatMult(mat, sol, res);
  VecAYPX(res, -1.0, rhs);
}

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms, int dim, int dofsPerNode, bool print) {
  /*
     int numSmoothIters = 3*dofsPerNode;
     if(dim > 1) {
     numSmoothIters *= 3;
     }
     if(dim > 2) {
     numSmoothIters *= 3;
     }
     */
  int numSmoothIters = 2;
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
        //KSPSetType(ksp[lev], KSPCG);
        KSPSetType(ksp[lev], KSPRICHARDSON);
        if(dim == 1) {
          KSPRichardsonSetScale(ksp[lev], (2.0/3.0));
        } else if (dim == 2) {
          KSPRichardsonSetScale(ksp[lev], (4.0/5.0));
        } else {
          KSPRichardsonSetScale(ksp[lev], (8.0/9.0));
        }
        KSPSetPreconditionerSide(ksp[lev], PC_LEFT);
        PCSetType(pc, PCJACOBI);
        //PCSetType(pc, PCSOR);
        //PCSORSetOmega(pc, 1.0);
        //PCSORSetSymmetric(pc, SOR_LOCAL_SYMMETRIC_SWEEP);
        //PCSORSetIterations(pc, 1, 1);
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
    if( (currNx < 17) || ((currNx%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
    if(dim > 1) {
      if( (currNy < 17) || ((currNy%2) == 0) ) {
        break;
      }
      currNy = 1 + ((currNy - 1)/2); 
    }
    if(dim > 2) {
      if( (currNz < 17) || ((currNz%2) == 0) ) {
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



