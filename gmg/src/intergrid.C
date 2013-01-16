
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void applyRestriction(Mat Pmat, Vec tmpCvec, Vec fVec, Vec cVec) {
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

void buildPmat(int dim, PetscInt dofsPerNode, std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec,
    std::vector<DM>& da, std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes) {
  Pmat.resize((da.size() - 1), NULL);
  tmpCvec.resize(Pmat.size(), NULL);
  for(int lev = 0; lev < (Pmat.size()); ++lev) {
    if(da[lev + 1] != NULL) {
      PetscInt xsf, ysf, zsf;
      PetscInt nxf, nyf, nzf;
      DMDAGetCorners(da[lev + 1], &xsf, &ysf, &zsf, &nxf, &nyf, &nzf);
      PetscInt xsc, ysc, zsc;
      PetscInt nxc, nyc, nzc;
      xsc = ysc = zsc = 0;
      nxc = nyc = nzc = 0;
      if(da[lev] != NULL) {
        DMDAGetCorners(da[lev], &xsc, &ysc, &zsc, &nxc, &nyc, &nzc);
      }
      if(dim < 3) {
        nzf = nzc = 1;
        zsf = zsc = 0;
      }
      if(dim < 2) {
        nyf = nyc = 1;
        ysf = ysc = 0;
      }
      PetscInt locRowSz = dofsPerNode*nxf*nyf*nzf;
      PetscInt locColSz = dofsPerNode*nxc*nyc*nzc;
      PetscInt* d_nnz = new PetscInt[locRowSz];
      PetscInt* o_nnz = NULL;
      if(activeNpes[lev + 1] > 1) {
        o_nnz = new PetscInt[locRowSz];
      }
      for(PetscInt zi = zsf, cnt = 0; zi < (zsf + nzf); ++zi) {
        bool oddZ = ((zi%2) != 0);
        std::vector<PetscInt> oz;
        oz.push_back((zi/2));
        if(oddZ) {
          oz.push_back((zi/2) + 1);
        }
        for(PetscInt yi = ysf; yi < (ysf + nyf); ++yi) {
          bool oddY = ((yi%2) != 0);
          std::vector<PetscInt> oy;
          oy.push_back((yi/2));
          if(oddY) {
            oy.push_back((yi/2) + 1);
          }
          for(PetscInt xi = xsf; xi < (xsf + nxf); ++xi) {
            bool oddX = ((xi%2) != 0);
            std::vector<PetscInt> ox;
            ox.push_back((xi/2));
            if(oddX) {
              ox.push_back((xi/2) + 1);
            }
            PetscInt diagVal = 0;
            PetscInt offVal = 0;
            for(size_t kk = 0; kk < oz.size(); ++kk) {
              for(size_t jj = 0; jj < oy.size(); ++jj) {
                for(size_t ii = 0; ii < ox.size(); ++ii) {
                  if((oz[kk] >= zsc) && (oz[kk] < (zsc + nzc)) &&  
                      (oy[jj] >= ysc) && (oy[jj] < (ysc + nyc)) &&
                      (ox[ii] >= xsc) && (ox[ii] < (xsc + nxc))) {
                    diagVal += dofsPerNode;
                  } else {
                    offVal += dofsPerNode;
                  }                
                }//end ii
              }//end jj
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
      MatCreate(activeComms[lev + 1], &(Pmat[lev]));
      MatSetSizes(Pmat[lev], locRowSz, locColSz, PETSC_DETERMINE, PETSC_DETERMINE);
      MatSetType(Pmat[lev], MATAIJ);
      if(activeNpes[lev + 1] > 1) {
        MatMPIAIJSetPreallocation(Pmat[lev], -1, d_nnz, -1, o_nnz);
      } else {
        MatSeqAIJSetPreallocation(Pmat[lev], -1, d_nnz);
      }
      delete [] d_nnz;
      if(activeNpes[lev + 1] > 1) {
        delete [] o_nnz;
      }
      MatGetVecs(Pmat[lev], &(tmpCvec[lev]), PETSC_NULL);
    }
  }//end lev
}

void computePmat(int dim, std::vector<unsigned long long int>& factorialsList, std::vector<Mat>& Pmat, 
    std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx, 
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<PetscInt> >& offsets,
    std::vector<std::vector<PetscInt> >& scanZ, std::vector<std::vector<PetscInt> >& scanY,
    std::vector<std::vector<PetscInt> >& scanX, PetscInt dofsPerNode,
    std::vector<long long int>& coeffs, const unsigned int K) {
  if(dim == 1) {
    for(int lev = 0; lev < (Pmat.size()); ++lev) {
      computePmat1D(factorialsList, Pmat[lev], Nx[lev], Nx[lev + 1], partX[lev], partX[lev + 1],
          offsets[lev], scanX[lev], offsets[lev + 1], scanX[lev + 1], dofsPerNode, coeffs, K); 
    }//end lev
  } else if(dim == 2) {
    for(int lev = 0; lev < (Pmat.size()); ++lev) {
      computePmat2D(factorialsList, Pmat[lev], Ny[lev], Nx[lev], Ny[lev + 1], Nx[lev + 1],
          partY[lev], partX[lev], partY[lev + 1], partX[lev + 1], offsets[lev], scanY[lev],
          scanX[lev], offsets[lev + 1], scanY[lev + 1], scanX[lev + 1], dofsPerNode, coeffs, K); 
    }//end lev
  } else {
    for(int lev = 0; lev < (Pmat.size()); ++lev) {
      computePmat3D(factorialsList, Pmat[lev], Nz[lev], Ny[lev], Nx[lev],
          Nz[lev + 1], Ny[lev + 1], Nx[lev + 1],
          partZ[lev], partY[lev], partX[lev],
          partZ[lev + 1], partY[lev + 1], partX[lev + 1],
          offsets[lev], scanZ[lev], scanY[lev], scanX[lev],
          offsets[lev + 1], scanZ[lev + 1], scanY[lev + 1], scanX[lev + 1],
          dofsPerNode, coeffs, K); 
    }//end lev
  }
}

void computePmat1D(std::vector<unsigned long long int>& factorialsList, Mat Pmat,
    PetscInt Nxc, PetscInt Nxf, std::vector<PetscInt>& partXc, std::vector<PetscInt>& partXf,
    std::vector<PetscInt>& cOffsets, std::vector<PetscInt>& scanXc,
    std::vector<PetscInt>& fOffsets, std::vector<PetscInt>& scanXf,
    PetscInt dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int fpi = rank;

  PetscInt fnx = partXf[fpi];

  PetscInt fxs = 0;
  if(fpi > 0) {
    fxs = 1 + scanXf[fpi - 1];
  }

  PetscInt fOff = fOffsets[rank];

  std::vector<std::vector<std::vector<long double> > > eval1Dderivatives(2);
  for(int nodeId = 0; nodeId < 2; ++nodeId) {
    eval1Dderivatives[nodeId].resize(K + 1);
    for(unsigned int cdof = 0; cdof <= K; ++cdof) {
      eval1Dderivatives[nodeId][cdof].resize(K + 1);
      for(unsigned int fdof = 0; fdof <= K; ++fdof) {
        eval1Dderivatives[nodeId][cdof][fdof] = eval1DshFnDerivative(factorialsList, 
            nodeId, cdof, K, coeffs, 0.0, fdof);
      }//end fdof
    }//end cdof
  }//end nodeId

  MatZeroEntries(Pmat);

  for(PetscInt fxi = fxs; fxi < (fxs + fnx); ++fxi) {
    PetscInt cxi = fxi/2;
    bool oddX = ((fxi%2) != 0);
    std::vector<PetscInt>::iterator xIt = std::lower_bound(scanXc.begin(), scanXc.end(), cxi);
    std::vector<PetscInt> xVec;
    std::vector<int> xPid;
    xVec.push_back(cxi);
    xPid.push_back((xIt - scanXc.begin()));
    if(oddX) {
      xVec.push_back(cxi + 1);
      if((*xIt) == cxi) {
        xPid.push_back((xIt - scanXc.begin() + 1));
      } else {
        xPid.push_back((xIt - scanXc.begin()));
      }
    }
    PetscInt fLoc = fxi - fxs;
    for(PetscInt xfd = 0; xfd <= K; ++xfd) {
      if((xfd == 0) && ((fxi == 0) || (fxi == (Nxf - 1)))) {
        continue;
      }
      PetscInt rowId = ((fOff + fLoc)*dofsPerNode) + xfd;
      for(size_t i = 0; i < xVec.size(); ++i) {
        PetscInt xLoc;
        if(xPid[i] > 0) {
          xLoc = xVec[i] - (1 + scanXc[xPid[i] - 1]);
        } else {
          xLoc = xVec[i];
        }
        int cPid = xPid[i];
        PetscInt cLoc = xLoc;
        for(PetscInt xcd = 0; xcd <= K; ++xcd) {
          if((xcd == 0) && ((xVec[i] == 0) || (xVec[i] == (Nxc - 1)))) {
            continue;
          }
          PetscInt colId = ((cOffsets[cPid] + cLoc)*dofsPerNode) + xcd;
          int xNodeId;
          if( (xVec[i] == (Nxc - 1)) || i ) {
            xNodeId = 1;
          } else {
            xNodeId = 0;
          }
          long double valX;
          if(oddX) {
            valX = eval1Dderivatives[xNodeId][xcd][xfd];
          } else {
            if(xcd == xfd) {
              valX = 1;
            } else {
              valX = 0;
            }
          }
          long double val = valX;
          unsigned long long int facExp = xfd;
          PetscScalar entry = val/(static_cast<long double>(1ull << facExp));
          MatSetValues(Pmat, 1, &rowId, 1, &colId, &entry, INSERT_VALUES);
        }//end xcd
      }//end i
    }//end xfd
  }//end fxi

  MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY);
}

void computePmat2D(std::vector<unsigned long long int>& factorialsList,
    Mat Pmat, PetscInt Nyc, PetscInt Nxc, PetscInt Nyf, PetscInt Nxf,
    std::vector<PetscInt>& partYc, std::vector<PetscInt>& partXc,
    std::vector<PetscInt>& partYf, std::vector<PetscInt>& partXf, 
    std::vector<PetscInt>& cOffsets, std::vector<PetscInt>& scanYc, std::vector<PetscInt>& scanXc,
    std::vector<PetscInt>& fOffsets, std::vector<PetscInt>& scanYf, std::vector<PetscInt>& scanXf,
    PetscInt dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int fpx = partXf.size();

  int fpj = rank/fpx;
  int fpi = rank%fpx;

  PetscInt fny = partYf[fpj];
  PetscInt fnx = partXf[fpi];

  PetscInt fys = 0;
  if(fpj > 0) {
    fys = 1 + scanYf[fpj - 1];
  }

  PetscInt fxs = 0;
  if(fpi > 0) {
    fxs = 1 + scanXf[fpi - 1];
  }

  PetscInt fOff = fOffsets[rank];

  int cpx = partXc.size();

  std::vector<std::vector<std::vector<long double> > > eval1Dderivatives(2);
  for(int nodeId = 0; nodeId < 2; ++nodeId) {
    eval1Dderivatives[nodeId].resize(K + 1);
    for(unsigned int cdof = 0; cdof <= K; ++cdof) {
      eval1Dderivatives[nodeId][cdof].resize(K + 1);
      for(unsigned int fdof = 0; fdof <= K; ++fdof) {
        eval1Dderivatives[nodeId][cdof][fdof] = eval1DshFnDerivative(factorialsList, 
            nodeId, cdof, K, coeffs, 0.0, fdof);
      }//end fdof
    }//end cdof
  }//end nodeId

  MatZeroEntries(Pmat);

  for(PetscInt fyi = fys; fyi < (fys + fny); ++fyi) {
    PetscInt cyi = fyi/2;
    bool oddY = ((fyi%2) != 0);
    std::vector<PetscInt>::iterator yIt = std::lower_bound(scanYc.begin(), scanYc.end(), cyi);
    std::vector<PetscInt> yVec;
    std::vector<int> yPid;
    yVec.push_back(cyi);
    yPid.push_back((yIt - scanYc.begin()));
    if(oddY) {
      yVec.push_back(cyi + 1);
      if((*yIt) == cyi) {
        yPid.push_back((yIt - scanYc.begin() + 1));
      } else {
        yPid.push_back((yIt - scanYc.begin()));
      }
    }
    for(PetscInt fxi = fxs; fxi < (fxs + fnx); ++fxi) {
      PetscInt cxi = fxi/2;
      bool oddX = ((fxi%2) != 0);
      std::vector<PetscInt>::iterator xIt = std::lower_bound(scanXc.begin(), scanXc.end(), cxi);
      std::vector<PetscInt> xVec;
      std::vector<int> xPid;
      xVec.push_back(cxi);
      xPid.push_back((xIt - scanXc.begin()));
      if(oddX) {
        xVec.push_back(cxi + 1);
        if((*xIt) == cxi) {
          xPid.push_back((xIt - scanXc.begin() + 1));
        } else {
          xPid.push_back((xIt - scanXc.begin()));
        }
      }
      PetscInt fLoc = ((fyi - fys)*fnx) + (fxi - fxs);
      for(PetscInt yfd = 0; yfd <= K; ++yfd) {
        if((yfd == 0) && ((fyi == 0) || (fyi == (Nyf - 1)))) {
          continue;
        }
        for(PetscInt xfd = 0; xfd <= K; ++xfd) {
          if((xfd == 0) && ((fxi == 0) || (fxi == (Nxf - 1)))) {
            continue;
          }
          PetscInt fd = (yfd*(K + 1)) + xfd;
          PetscInt rowId = ((fOff + fLoc)*dofsPerNode) + fd;
          for(size_t j = 0; j < yVec.size(); ++j) {
            PetscInt yLoc;
            if(yPid[j] > 0) {
              yLoc = yVec[j] - (1 + scanYc[yPid[j] - 1]);
            } else {
              yLoc = yVec[j];
            }
            for(size_t i = 0; i < xVec.size(); ++i) {
              PetscInt xLoc;
              if(xPid[i] > 0) {
                xLoc = xVec[i] - (1 + scanXc[xPid[i] - 1]);
              } else {
                xLoc = xVec[i];
              }
              int cPid = (yPid[j]*cpx) + xPid[i];
              PetscInt cLoc = (yLoc*partXc[xPid[i]]) + xLoc;
              for(PetscInt ycd = 0; ycd <= K; ++ycd) {
                if((ycd == 0) && ((yVec[j] == 0) || (yVec[j] == (Nyc - 1)))) {
                  continue;
                }
                for(PetscInt xcd = 0; xcd <= K; ++xcd) {
                  if((xcd == 0) && ((xVec[i] == 0) || (xVec[i] == (Nxc - 1)))) {
                    continue;
                  }
                  PetscInt cd = (ycd*(K + 1)) + xcd;
                  PetscInt colId = ((cOffsets[cPid] + cLoc)*dofsPerNode) + cd;
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
                  long double valX;
                  if(oddX) {
                    valX = eval1Dderivatives[xNodeId][xcd][xfd];
                  } else {
                    if(xcd == xfd) {
                      valX = 1;
                    } else {
                      valX = 0;
                    }
                  }
                  long double valY;
                  if(oddY) {
                    valY = eval1Dderivatives[yNodeId][ycd][yfd];
                  } else {
                    if(ycd == yfd) {
                      valY = 1;
                    } else {
                      valY = 0;
                    }
                  }
                  long double val = valX * valY;
                  unsigned long long int facExp = xfd + yfd;
                  PetscScalar entry = val/(static_cast<long double>(1ull << facExp));
                  MatSetValues(Pmat, 1, &rowId, 1, &colId, &entry, INSERT_VALUES);
                }//end xcd
              }//end ycd
            }//end i
          }//end j
        }//end xfd
      }//end yfd
    }//end fxi
  }//end fyi

  MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY);
}

void computePmat3D(std::vector<unsigned long long int>& factorialsList,
    Mat Pmat, PetscInt Nzc, PetscInt Nyc, PetscInt Nxc, PetscInt Nzf, PetscInt Nyf, PetscInt Nxf,
    std::vector<PetscInt>& partZc, std::vector<PetscInt>& partYc, std::vector<PetscInt>& partXc,
    std::vector<PetscInt>& partZf, std::vector<PetscInt>& partYf, std::vector<PetscInt>& partXf, 
    std::vector<PetscInt>& cOffsets, std::vector<PetscInt>& scanZc,
    std::vector<PetscInt>& scanYc, std::vector<PetscInt>& scanXc,
    std::vector<PetscInt>& fOffsets, std::vector<PetscInt>& scanZf, 
    std::vector<PetscInt>& scanYf, std::vector<PetscInt>& scanXf,
    PetscInt dofsPerNode, std::vector<long long int>& coeffs, const unsigned int K) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int fpx = partXf.size();
  int fpy = partYf.size();

  int fpk = rank/(fpx*fpy);
  int fpj = (rank/fpx)%fpy;
  int fpi = rank%fpx;

  PetscInt fnz = partZf[fpk];
  PetscInt fny = partYf[fpj];
  PetscInt fnx = partXf[fpi];

  PetscInt fzs = 0;
  if(fpk > 0) {
    fzs = 1 + scanZf[fpk - 1];
  }

  PetscInt fys = 0;
  if(fpj > 0) {
    fys = 1 + scanYf[fpj - 1];
  }

  PetscInt fxs = 0;
  if(fpi > 0) {
    fxs = 1 + scanXf[fpi - 1];
  }

  PetscInt fOff = fOffsets[rank];

  int cpx = partXc.size();
  int cpy = partYc.size();

  std::vector<std::vector<std::vector<long double> > > eval1Dderivatives(2);
  for(int nodeId = 0; nodeId < 2; ++nodeId) {
    eval1Dderivatives[nodeId].resize(K + 1);
    for(unsigned int cdof = 0; cdof <= K; ++cdof) {
      eval1Dderivatives[nodeId][cdof].resize(K + 1);
      for(unsigned int fdof = 0; fdof <= K; ++fdof) {
        eval1Dderivatives[nodeId][cdof][fdof] = eval1DshFnDerivative(factorialsList, 
            nodeId, cdof, K, coeffs, 0.0, fdof);
      }//end fdof
    }//end cdof
  }//end nodeId

  MatZeroEntries(Pmat);

  for(PetscInt fzi = fzs; fzi < (fzs + fnz); ++fzi) {
    PetscInt czi = fzi/2;
    bool oddZ = ((fzi%2) != 0);
    std::vector<PetscInt>::iterator zIt = std::lower_bound(scanZc.begin(), scanZc.end(), czi);
    std::vector<PetscInt> zVec;
    std::vector<int> zPid;
    zVec.push_back(czi);
    zPid.push_back((zIt - scanZc.begin()));
    if(oddZ) {
      zVec.push_back(czi + 1);
      if((*zIt) == czi) {
        zPid.push_back((zIt - scanZc.begin() + 1));
      } else {
        zPid.push_back((zIt - scanZc.begin()));
      }
    }
    for(PetscInt fyi = fys; fyi < (fys + fny); ++fyi) {
      PetscInt cyi = fyi/2;
      bool oddY = ((fyi%2) != 0);
      std::vector<PetscInt>::iterator yIt = std::lower_bound(scanYc.begin(), scanYc.end(), cyi);
      std::vector<PetscInt> yVec;
      std::vector<int> yPid;
      yVec.push_back(cyi);
      yPid.push_back((yIt - scanYc.begin()));
      if(oddY) {
        yVec.push_back(cyi + 1);
        if((*yIt) == cyi) {
          yPid.push_back((yIt - scanYc.begin() + 1));
        } else {
          yPid.push_back((yIt - scanYc.begin()));
        }
      }
      for(PetscInt fxi = fxs; fxi < (fxs + fnx); ++fxi) {
        PetscInt cxi = fxi/2;
        bool oddX = ((fxi%2) != 0);
        std::vector<PetscInt>::iterator xIt = std::lower_bound(scanXc.begin(), scanXc.end(), cxi);
        std::vector<PetscInt> xVec;
        std::vector<int> xPid;
        xVec.push_back(cxi);
        xPid.push_back((xIt - scanXc.begin()));
        if(oddX) {
          xVec.push_back(cxi + 1);
          if((*xIt) == cxi) {
            xPid.push_back((xIt - scanXc.begin() + 1));
          } else {
            xPid.push_back((xIt - scanXc.begin()));
          }
        }
        PetscInt fLoc = ((((fzi - fzs)*fny) + (fyi - fys))*fnx) + (fxi - fxs);
        for(PetscInt zfd = 0; zfd <= K; ++zfd) {
          if((zfd == 0) && ((fzi == 0) || (fzi == (Nzf - 1)))) {
            continue;
          }
          for(PetscInt yfd = 0; yfd <= K; ++yfd) {
            if((yfd == 0) && ((fyi == 0) || (fyi == (Nyf - 1)))) {
              continue;
            }
            for(PetscInt xfd = 0; xfd <= K; ++xfd) {
              if((xfd == 0) && ((fxi == 0) || (fxi == (Nxf - 1)))) {
                continue;
              }
              PetscInt fd = (((zfd*(K + 1)) + yfd)*(K + 1)) + xfd;
              PetscInt rowId = ((fOff + fLoc)*dofsPerNode) + fd;
              for(size_t k = 0; k < zVec.size(); ++k) {
                PetscInt zLoc;
                if(zPid[k] > 0) {
                  zLoc = zVec[k] - (1 + scanZc[zPid[k] - 1]);
                } else {
                  zLoc = zVec[k];
                }
                for(size_t j = 0; j < yVec.size(); ++j) {
                  PetscInt yLoc;
                  if(yPid[j] > 0) {
                    yLoc = yVec[j] - (1 + scanYc[yPid[j] - 1]);
                  } else {
                    yLoc = yVec[j];
                  }
                  for(size_t i = 0; i < xVec.size(); ++i) {
                    PetscInt xLoc;
                    if(xPid[i] > 0) {
                      xLoc = xVec[i] - (1 + scanXc[xPid[i] - 1]);
                    } else {
                      xLoc = xVec[i];
                    }
                    int cPid = (((zPid[k]*cpy) + yPid[j])*cpx) + xPid[i];
                    PetscInt cLoc = (((zLoc*partYc[yPid[j]]) + yLoc)*partXc[xPid[i]]) + xLoc;
                    for(PetscInt zcd = 0; zcd <= K; ++zcd) {
                      if((zcd == 0) && ((zVec[k] == 0) || (zVec[k] == (Nzc - 1)))) {
                        continue;
                      }
                      for(PetscInt ycd = 0; ycd <= K; ++ycd) {
                        if((ycd == 0) && ((yVec[j] == 0) || (yVec[j] == (Nyc - 1)))) {
                          continue;
                        }
                        for(PetscInt xcd = 0; xcd <= K; ++xcd) {
                          if((xcd == 0) && ((xVec[i] == 0) || (xVec[i] == (Nxc - 1)))) {
                            continue;
                          }
                          PetscInt cd = (((zcd*(K + 1)) + ycd)*(K + 1)) + xcd;
                          PetscInt colId = ((cOffsets[cPid] + cLoc)*dofsPerNode) + cd;
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
                          long double valX;
                          if(oddX) {
                            valX = eval1Dderivatives[xNodeId][xcd][xfd];
                          } else {
                            if(xcd == xfd) {
                              valX = 1;
                            } else {
                              valX = 0;
                            }
                          }
                          long double valY;
                          if(oddY) {
                            valY = eval1Dderivatives[yNodeId][ycd][yfd];
                          } else {
                            if(ycd == yfd) {
                              valY = 1;
                            } else {
                              valY = 0;
                            }
                          }
                          long double valZ;
                          if(oddZ) {
                            valZ = eval1Dderivatives[zNodeId][zcd][zfd];
                          } else {
                            if(zcd == zfd) {
                              valZ = 1;
                            } else {
                              valZ = 0;
                            }
                          }
                          long double val = valX * valY * valZ;
                          unsigned long long int facExp = xfd + yfd + zfd;
                          PetscScalar entry = val/(static_cast<long double>(1ull << facExp));
                          MatSetValues(Pmat, 1, &rowId, 1, &colId, &entry, INSERT_VALUES);
                        }//end xcd
                      }//end ycd
                    }//end zcd
                  }//end i
                }//end j
              }//end k
            }//end xfd
          }//end yfd
        }//end zfd
      }//end fxi
    }//end fyi
  }//end fzi

  MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY);
}


