
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

extern PetscLogEvent buildPmatEvent;

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

void buildPmat(std::vector<unsigned long long int>& factorialsList, 
    std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, 
    std::vector<int>& activeNpes, int dim, int dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K, 
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, std::vector<std::vector<PetscInt> >& partZ,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<int> >& offsets,
    std::vector<std::vector<int> >& scanLz, std::vector<std::vector<int> >& scanLy, std::vector<std::vector<int> >& scanLx, bool print) {
  PetscLogEventBegin(buildPmatEvent, 0, 0, 0, 0);

  Pmat.resize((da.size() - 1), NULL);
  tmpCvec.resize(Pmat.size(), NULL);
  for(int lev = 0; lev < (Pmat.size()); ++lev) {
    if(da[lev + 1] != NULL) {
      PetscInt nxf, nyf, nzf;
      DMDAGetCorners(da[lev + 1], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nxf, &nyf, &nzf);
      MatCreate(activeComms[lev + 1], &(Pmat[lev]));
      PetscInt nxc, nyc, nzc;
      nxc = nyc = nzc = 0;
      if(da[lev] != NULL) {
        DMDAGetCorners(da[lev], PETSC_NULL, PETSC_NULL, PETSC_NULL, &nxc, &nyc, &nzc);
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
          PetscInt rowId = ((fOff + fLoc)*dofsPerNode) + fd;
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
                  PetscInt colId = ((cOffsets[cPid] + cLoc)*dofsPerNode) + d;
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
                  long double val = eval1Dderivatives[xNodeId][xcd][xPtId][xfd];
                  unsigned long long int facExp = xfd;
                  if(dim > 1) {
                    val *= eval1Dderivatives[yNodeId][ycd][yPtId][yfd];
                    facExp += yfd;
                  } 
                  if(dim > 2) {
                    val *= eval1Dderivatives[zNodeId][zcd][zPtId][zfd];
                    facExp += zfd;
                  }
                  PetscScalar val2 = val/(static_cast<long double>(1ull << facExp));
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
}



