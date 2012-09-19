
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"
#include "petscmg.h"

extern PetscLogEvent buildPmatEvent;
extern PetscLogEvent buildKmatEvent;
extern PetscLogEvent vCycleEvent;

void buildPmat(std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DA>& da,
    std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dim, int dofsPerNode,
    std::vector<long long int>& coeffs, const unsigned int K, std::vector<PetscInt> & Nz, 
    std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, std::vector<std::vector<PetscInt> >& partZ,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, bool print) {
  PetscLogEventBegin(buildPmatEvent, 0, 0, 0, 0);

  Pmat.resize((da.size() - 1), NULL);
  tmpCvec.resize(Pmat.size(), NULL);
  for(int lev = 0; lev < (Pmat.size()); ++lev) {
    if(da[lev + 1] != NULL) {
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
      //PERFORMANCE IMPROVEMENT: Better PreAllocation.
      if(activeNpes[lev + 1] > 1) {
        MatMPIAIJSetPreallocation(Pmat[lev], (dofsPerElem*dofsPerNode), PETSC_NULL, (dofsPerElem*dofsPerNode), PETSC_NULL);
      } else {
        MatSeqAIJSetPreallocation(Pmat[lev], (dofsPerElem*dofsPerNode), PETSC_NULL);
      }
      MatGetVecs(Pmat[lev], &(tmpCvec[lev]), PETSC_NULL);
      computePmat(Pmat[lev], Nz[lev], Ny[lev], Nx[lev], Nz[lev + 1], Ny[lev + 1], Nx[lev + 1],
          partZ[lev], partY[lev], partX[lev], partZ[lev + 1], partY[lev + 1], partX[lev + 1],
          dim, dofsPerNode, coeffs, K);
    }
    if(print) {
      std::cout<<"Built Pmat for level = "<<lev<<std::endl;
    }
  }//end lev

  PetscLogEventEnd(buildPmatEvent, 0, 0, 0, 0);
}

void computePmat(Mat Pmat, int Nzc, int Nyc, int Nxc, int Nzf, int Nyf, int Nxf,
    std::vector<PetscInt>& lzc, std::vector<PetscInt>& lyc, std::vector<PetscInt>& lxc,
    std::vector<PetscInt>& lzf, std::vector<PetscInt>& lyf, std::vector<PetscInt>& lxf,
    int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K) {
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
  for(int k = 0; k < fpk; ++k) {
    fzs += lzf[k];
  }//end k

  int fys = 0;
  for(int j = 0; j < fpj; ++j) {
    fys += lyf[j];
  }//end j

  int fxs = 0;
  for(int i = 0; i < fpi; ++i) {
    fxs += lxf[i];
  }//end i

  int fOffset = 0;
  for(int p = 0; p < rank; ++p) {
    int k = p/(fpx*fpy);
    int j = (p/fpx)%fpy;
    int i = p%fpx;
    fOffset += (lzf[k]*lyf[j]*lxf[i]);
  }//end p

  int cpx = lxc.size();
  int cpy = lyc.size();
  int cpz = lzc.size();

  int cNpes = cpx*cpy*cpz;

  std::vector<int> cOffsets(cNpes);
  cOffsets[0] = 0;
  for(int p = 1; p < cNpes; ++p) {
    int k = (p - 1)/(cpx*cpy);
    int j = ((p - 1)/cpx)%cpy;
    int i = (p - 1)%cpx;
    cOffsets[p] = cOffsets[p - 1] + (lzc[k]*lyc[j]*lxc[i]);
  }//end p

  std::vector<int> scanClx(cpx);
  scanClx[0] = lxc[0] - 1;
  for(int i = 1; i < cpx; ++i) {
    scanClx[i] = scanClx[i - 1] + lxc[i];
  }//end i

  std::vector<int> scanCly(cpy);
  scanCly[0] = lyc[0] - 1;
  for(int i = 1; i < cpy; ++i) {
    scanCly[i] = scanCly[i - 1] + lyc[i];
  }//end i

  std::vector<int> scanClz(cpz);
  scanClz[0] = lzc[0] - 1;
  for(int i = 1; i < cpz; ++i) {
    scanClz[i] = scanClz[i - 1] + lzc[i];
  }//end i

  double hxf, hyf, hzf;
  double hxc, hyc, hzc;
  hxf = 1.0/(static_cast<double>(Nxf - 1));
  hxc = 1.0/(static_cast<double>(Nxc - 1));
  if(dim > 1) {
    hyf = 1.0/(static_cast<double>(Nyf - 1));
    hyc = 1.0/(static_cast<double>(Nyc - 1));
  } else {
    hyf = 1.0;
    hyc = 1.0;
  }
  if(dim > 2) {
    hzf = 1.0/(static_cast<double>(Nzf - 1));
    hzc = 1.0/(static_cast<double>(Nzc - 1));
  } else {
    hzf = 1.0;
    hzc = 1.0;
  }

  MatZeroEntries(Pmat);

  for(int fzi = fzs; fzi < (fzs + fnz); ++fzi) {
    int czi = fzi/2;
    bool oddZ = ((fzi%2) != 0);
    std::vector<int>::iterator zIt = std::lower_bound(scanClz.begin(), scanClz.end(), czi);
    assert(zIt != scanClz.end());
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
      assert(yIt != scanCly.end());
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
        assert(xIt != scanClx.end());
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
          //PERFORMANCE IMPROVEMENT: Pre-Compute 
          double factor = (std::pow((0.5*hzf), zfd))*(std::pow((0.5*hyf), yfd))*(std::pow((0.5*hxf), xfd));
          int rowId = ((fOffset + fLoc)*dofsPerNode) + fd;
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
                  double val;
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
                  double xPt;
                  if(oddX) {
                    xPt = 0.0;
                  } else {
                    if(xNodeId == 0) {
                      xPt = -1.0;
                    } else {
                      xPt = 1.0;
                    }
                  }
                  double yPt;
                  if(oddY) {
                    yPt = 0.0;
                  } else {
                    if(yNodeId == 0) {
                      yPt = -1.0;
                    } else {
                      yPt = 1.0;
                    }
                  }
                  double zPt;
                  if(oddZ) {
                    zPt = 0.0;
                  } else {
                    if(zNodeId == 0) {
                      zPt = -1.0;
                    } else {
                      zPt = 1.0;
                    }
                  }
                  //PERFORMANCE IMPROVEMENT: Pre-Compute 
                  if(dim == 1) {
                    val = eval1DshFnGderivative(xNodeId, xcd, K, 
                        coeffs, xPt, xfd, hxc);
                  } else if(dim == 2) {
                    val = eval2DshFnGderivative(yNodeId, xNodeId, ycd, 
                        xcd, K, coeffs, yPt, xPt, yfd, xfd, hyc, hxc);
                  } else {
                    val = eval3DshFnGderivative(zNodeId, yNodeId, xNodeId,
                        zcd, ycd, xcd, K, coeffs, zPt, yPt, xPt, 
                        zfd, yfd, xfd, hzc, hyc, hxc);
                  }
                  val *= factor;
                  MatSetValues(Pmat, 1, &rowId, 1, &colId, &val, INSERT_VALUES);
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
}

void buildKmat(std::vector<Mat>& Kmat, std::vector<DA>& da, std::vector<long long int>& coeffs, const unsigned int K, bool print) {
  PetscLogEventBegin(buildKmatEvent, 0, 0, 0, 0);

  Kmat.resize(da.size(), NULL);
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      DAGetMatrix(da[i], MATAIJ, &(Kmat[i]));
      if(i == 0) {
        computeKmat(Kmat[i], da[i], coeffs, K, print);
      } else {
        computeKmat(Kmat[i], da[i], coeffs, K, false);
      }
      dirichletMatrixCorrection(Kmat[i], da[i]);
    }
    if(print) {
      std::cout<<"Built Kmat for level = "<<i<<std::endl;
    }
  }//end i

  PetscLogEventEnd(buildKmatEvent, 0, 0, 0, 0);
}

void computeKmat(Mat Kmat, DA da, std::vector<long long int>& coeffs, const unsigned int K, bool print) {
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

  if(dim < 2) {
    Ny = 1;
    ys = 0;
    ny = 1;
  }
  if(dim < 3) {
    Nz = 1; 
    zs = 0;
    nz = 1;
  }

  double hx, hy, hz;
  hx = 1.0/(static_cast<double>(Nx - 1));
  if(dim > 1) {
    hy = 1.0/(static_cast<double>(Ny - 1));
  }
  if(dim > 2) {
    hz = 1.0/(static_cast<double>(Nz - 1));
  }

  std::vector<std::vector<double> > elemMat;
  if(dim == 1) {
    createPoisson1DelementMatrix(K, coeffs, hx, elemMat, print);
  } else if(dim == 2) {
    createPoisson2DelementMatrix(K, coeffs, hy, hx, elemMat, print);
  } else {
    createPoisson3DelementMatrix(K, coeffs, hz, hy, hx, elemMat, print);
  }

  PetscInt nxe = nx;
  PetscInt nye = ny;
  PetscInt nze = nz;

  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  }
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  }

  unsigned int nodesPerElem = (1 << dim);

  MatZeroEntries(Kmat);

  MatStencil row;
  MatStencil col;
  for(unsigned int zi = zs; zi < (zs + nze); ++zi) {
    for(unsigned int yi = ys; yi < (ys + nye); ++yi) {
      for(unsigned int xi = xs; xi < (xs + nxe); ++xi) {
        for(unsigned int nr = 0, r = 0; nr < nodesPerElem; ++nr) {
          unsigned int zr = (nr/4);
          unsigned int yr = ((nr/2)%2);
          unsigned int xr = (nr%2);
          for(unsigned int dr = 0; dr < dofsPerNode; ++r, ++dr) {
            row.k = zi + zr;
            row.j = yi + yr; 
            row.i = xi + xr;
            row.c = dr; 
            for(unsigned int nc = 0, c = 0; nc < nodesPerElem; ++nc) {
              unsigned int zc = (nc/4);
              unsigned int yc = ((nc/2)%2);
              unsigned int xc = (nc%2);
              for(unsigned int dc = 0; dc < dofsPerNode; ++c, ++dc) {
                col.k = zi + zc;
                col.j = yi + yc;
                col.i = xi + xc;
                col.c = dc;
                PetscScalar val = elemMat[r][c];
                MatSetValuesStencil(Kmat, 1, &row, 1, &col, &val, ADD_VALUES);
              }//end dc
            }//end nc
          }//end dr
        }//end nr
      }//end xi
    }//end yi
  }//end zi

  MatAssemblyBegin(Kmat, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FLUSH_ASSEMBLY);
}

void dirichletMatrixCorrection(Mat Kmat, DA da) {
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

  if(dim < 2) {
    Ny = 1;
    ys = 0;
    ny = 1;
  }
  if(dim < 3) {
    Nz = 1; 
    zs = 0;
    nz = 1;
  }

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
  }

  std::vector<PetscInt> zvec;
  if(dim > 2) {
    if(zs == 0) {
      zvec.push_back(0);
    }
    if((zs + nz) == Nz) {
      zvec.push_back((Nz - 1));
    }
  }

  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;

  MatStencil bnd;
  bnd.c = 0;

  MatStencil oth;

  //x
  for(int b = 0; b < xvec.size(); ++b) {
    bnd.i = xvec[b];
    for(int zi = zs; zi < (zs + nz); ++zi) {
      bnd.k = zi;
      for(int yi = ys; yi < (ys + ny); ++yi) {
        bnd.j = yi;
        for(int k = -1; k < 2; ++k) {
          oth.k = (bnd.k) + k;
          if( ((oth.k) >= 0) && ((oth.k) < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              oth.j = (bnd.j) + j;
              if( ((oth.j) >= 0) && ((oth.j) < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  oth.i = (bnd.i) + i;
                  if( ((oth.i) >= 0) && ((oth.i) < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
                      oth.c = d;
                      if(k || j || i || d) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
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
    bnd.j = yvec[b];
    for(int zi = zs; zi < (zs + nz); ++zi) {
      bnd.k = zi;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        bnd.i = xi; 
        for(int k = -1; k < 2; ++k) {
          oth.k = (bnd.k) + k;
          if( ((oth.k) >= 0) && ((oth.k) < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              oth.j = (bnd.j) + j;
              if( ((oth.j) >= 0) && ((oth.j) < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  oth.i = (bnd.i) + i;
                  if( ((oth.i) >= 0) && ((oth.i) < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
                      oth.c = d;
                      if(k || j || i || d) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
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
    bnd.k = zvec[b];
    for(int yi = ys; yi < (ys + ny); ++yi) {
      bnd.j = yi;
      for(int xi = xs; xi < (xs + nx); ++xi) {
        bnd.i = xi; 
        for(int k = -1; k < 2; ++k) {
          oth.k = (bnd.k) + k;
          if( ((oth.k) >= 0) && ((oth.k) < Nz) ) {
            for(int j = -1; j < 2; ++j) {
              oth.j = (bnd.j) + j;
              if( ((oth.j) >= 0) && ((oth.j) < Ny) ) {
                for(int i = -1; i < 2; ++i) {
                  oth.i = (bnd.i) + i;
                  if( ((oth.i) >= 0) && ((oth.i) < Nx) ) {
                    for(int d = 0; d < dofsPerNode; ++d) {
                      oth.c = d;
                      if(k || j || i || d) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
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

void computeRandomRHS(DA da, Mat Kmat, Vec rhs, const unsigned int seed) {
  PetscRandom rndCtx;
  PetscRandomCreate(MPI_COMM_WORLD, &rndCtx);
  PetscRandomSetType(rndCtx, PETSCRAND48);
  PetscRandomSetSeed(rndCtx, seed);
  PetscRandomSeed(rndCtx);
  Vec tmpSol;
  VecDuplicate(rhs, &tmpSol);
  //VecSetRandom(tmpSol, rndCtx);
  VecSet(tmpSol, 10.0);
  PetscRandomDestroy(rndCtx);
  zeroBoundaries(da, tmpSol);
  assert(Kmat != NULL);
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

  assert(ksp[currLev] != NULL);
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
  assert(Pmat != NULL);
  assert(fVec != NULL);
  assert(tmpCvec != NULL);
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
  assert(Pmat != NULL);
  assert(fVec != NULL);
  assert(tmpCvec != NULL);
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
  int numSmoothIters = 2*dofsPerNode;
  if(dim > 1) {
    numSmoothIters *= 2;
  }
  if(dim > 2) {
    numSmoothIters *= 2;
  }
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
        KSPSetType(ksp[lev], KSPRICHARDSON);
        //KSPSetType(ksp[lev], KSPCG);
        KSPRichardsonSetScale(ksp[lev], 1.0);
        KSPSetPreconditionerSide(ksp[lev], PC_LEFT);
        //PCSetType(pc, PCJACOBI);
        PCSetType(pc, PCSOR);
        PCSORSetOmega(pc, 1.0);
        PCSORSetSymmetric(pc, SOR_LOCAL_SYMMETRIC_SWEEP);
        PCSORSetIterations(pc, 1, 1);
        KSPSetInitialGuessNonzero(ksp[lev], PETSC_TRUE);
      }
      KSPSetOperators(ksp[lev], Kmat[lev], Kmat[lev], SAME_NONZERO_PATTERN);
      KSPSetTolerances(ksp[lev], 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
    }
  }//end lev
}

void createDA(std::vector<DA>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dofsPerNode,
    int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, MPI_Comm globalComm, bool print) {
  int globalRank;
  int globalNpes;
  MPI_Comm_rank(globalComm, &globalRank);
  MPI_Comm_size(globalComm, &globalNpes);

  int maxCoarseNpes = globalNpes;
  PetscOptionsGetInt(PETSC_NULL, "-maxCoarseNpes", &maxCoarseNpes, PETSC_NULL);
  if(maxCoarseNpes > globalNpes) {
    maxCoarseNpes = globalNpes;
  }
  assert(maxCoarseNpes > 0);

  int numLevels = Nx.size();
  assert(numLevels > 0);
  activeNpes.resize(numLevels);
  activeComms.resize(numLevels);
  da.resize(numLevels);
  partZ.resize(numLevels);
  partY.resize(numLevels);
  partX.resize(numLevels);

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
    computePartition(dim, Nz[lev], Ny[lev], Nx[lev], maxNpes, partZ[lev], partY[lev], partX[lev]);
    PetscInt pz = (partZ[lev]).size();
    PetscInt py = (partY[lev]).size();
    PetscInt px = (partX[lev]).size();
    activeNpes[lev] = (px*py*pz);
    if(print) {
      std::cout<<"Active Npes for Level "<<lev<<" = "<<(activeNpes[lev])
        <<" : (px, py, pz) = ("<<px<<", "<<py<<", "<<pz<<")"<<std::endl;
    }
    if(lev > 0) {
      assert(activeNpes[lev] >= activeNpes[lev - 1]);
    }
    if(globalRank < (activeNpes[lev])) {
      MPI_Group subGroup;
      MPI_Group_incl(globalGroup, (activeNpes[lev]), rankList, &subGroup);
      MPI_Comm_create(globalComm, subGroup, &(activeComms[lev]));
      MPI_Group_free(&subGroup);
      DACreate(activeComms[lev], dim, DA_NONPERIODIC, DA_STENCIL_BOX, (Nx[lev]), (Ny[lev]), (Nz[lev]),
          px, py, pz, dofsPerNode, 1, &(partX[lev][0]), &(partY[lev][0]), &(partZ[lev][0]), (&(da[lev])));
    } else {
      MPI_Comm_create(globalComm, MPI_GROUP_EMPTY, &(activeComms[lev]));
      assert(activeComms[lev] == MPI_COMM_NULL);
      da[lev] = NULL;
    }
  }//end lev

  delete [] rankList;
  MPI_Group_free(&globalGroup);
}

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes,
    std::vector<PetscInt> &lz, std::vector<PetscInt> &ly, std::vector<PetscInt> &lx) {
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
  assert(((pList[0])*(pList[1])*(pList[2])) <= maxNpes);

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

  int pz;
  assert(Nz == Nlist[0]);
  pz = pList[0];

  assert((px*py*pz) <= maxNpes);
  assert(px >= 1);
  assert(py >= 1);
  assert(pz >= 1);
  assert(px <= Nx);
  assert(py <= Ny);
  assert(pz <= Nz);

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
}

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, bool print) {
  PetscInt currNx = 17;
  PetscInt currNy = 1;
  PetscInt currNz = 1;

  assert(dim > 0);
  assert(dim <= 3);

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
    if( (currNx < 9) || ((currNx%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
    if(dim > 1) {
      if( (currNy < 9) || ((currNy%2) == 0) ) {
        break;
      }
      currNy = 1 + ((currNy - 1)/2); 
    }
    if(dim > 2) {
      if( (currNz < 9) || ((currNz%2) == 0) ) {
        break;
      }
      currNz = 1 + ((currNz - 1)/2); 
    }
  }//lev

  if(dim < 2) {
    assert(Ny.empty());
    Ny.resize((Nx.size()), 1);
  } else { 
    assert( (Ny.size()) == (Nx.size()) );
  }

  if(dim < 3) {
    assert(Nz.empty());
    Nz.resize((Nx.size()), 1);
  } else {
    assert( (Nz.size()) == (Nx.size()) );
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



