
PetscErrorCode applyPCFD1D(PC pc, Vec in, Vec out) {
  PCFD1Ddata* data;
  PCShellGetContext(pc, (void**)(&data));

  MPI_Comm comm;
  PetscObjectGetComm((PetscObject)pc, &comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  int nx = (*(data->partX))[rank];
  int numDofs = data->numDofs;

  double* inArr;
  double* rhsArr;
  VecGetArray((data->rhs), &rhsArr);
  VecGetArray(in, &inArr);
  for(int i = 0; i < nx; ++i) {
    for(int d = 0; d < (numDofs - 1); ++d) {
      rhsArr[((numDofs - 1) * i) + d] = inArr[(numDofs * i) + d];
    }//end d
  }//end i
  VecRestoreArray((data->rhs), &rhsArr);
  VecRestoreArray(in, &inArr);

  KSPSolve((data->ksp), (data->rhs), (data->sol));

  double* outArr;
  VecGetArray(out, &outArr);

  double* solArr;
  double* uArr;
  VecGetArray((data->sol), &solArr);
  VecGetArray((data->u), &uArr);
  for(int i = 0; i < nx; ++i) {
    for(int d = 0; d < (numDofs - 1); ++d) {
      outArr[(numDofs * i) + d] = solArr[((numDofs - 1) * i) + d];
    }//end d
    uArr[i] = solArr[((numDofs - 1) * i) + (numDofs - 2)];
  }//end i
  VecRestoreArray((data->sol), &solArr);
  VecRestoreArray((data->u), &uArr);

  applyFD1D(comm, *(data->partX), (data->u), (data->uPrime));

  double* uPrimeArr;
  VecGetArray((data->uPrime), &uPrimeArr);
  for(int i = 0; i < nx; ++i) {
    outArr[(numDofs * i) + (numDofs - 1)] = uPrimeArr[i];
  }//end i
  VecRestoreArray((data->uPrime), &uPrimeArr);

  VecRestoreArray(out, &outArr);

  return 0;
}

PetscErrorCode Khat1Dmult(Mat mat, Vec in, Vec out) {
  Khat1Ddata* data;
  MatShellGetContext(mat, &data);

  MPI_Comm comm;
  PetscObjectGetComm((PetscObject)mat, &comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  int nx = (*(data->partX))[rank];
  int numDofs = data->numDofs;
  int K = data->K;

  VecZeroEntries(out);

  double* outArr;
  VecGetArray(out, &outArr);

  double* inArr;
  VecGetArray(in, &inArr);
  for(int c = 0; c < numDofs; ++c) {
    double* uArr;
    VecGetArray((data->u), &uArr);
    for(int i = 0; i < nx; ++i) {
      uArr[i] = inArr[(numDofs * i) + c];
    }//end i
    VecRestoreArray((data->u), &uArr);

    for(int r = 0; r < numDofs; ++r) {
      if(r <= c) {
        MatMult(((*(data->blkKmats))[r][c - r]), (data->u), (data->tmpOut));
      } else {
        MatMultTranspose(((*(data->blkKmats))[c][r - c]), (data->u), (data->tmpOut));
      }
      double* tmpArr;
      VecGetArray((data->tmpOut), &tmpArr);
      for(int i = 0; i < nx; ++i) {
        outArr[(numDofs * i) + r] += tmpArr[i];
      }//end i
      VecRestoreArray((data->tmpOut), &tmpArr);
    }//end r
  }//end c
  VecRestoreArray(in, &inArr);

  for(int c = numDofs; c <= K; ++c) {
    applyFD1D(comm, *(data->partX), (data->u), (data->uPrime));
    for(int r = 0; r < numDofs; ++r) {
      MatMult(((*(data->blkKmats))[r][c - r]), (data->uPrime), (data->tmpOut));
      double* tmpArr;
      VecGetArray((data->tmpOut), &tmpArr);
      for(int i = 0; i < nx; ++i) {
        outArr[(numDofs * i) + r] += tmpArr[i];
      }//end i
      VecRestoreArray((data->tmpOut), &tmpArr);
    }//end r
    VecSwap((data->u), (data->uPrime));
  }//end c

  VecRestoreArray(out, &outArr);

  return 0;
}

void create1DmatShells(MPI_Comm comm, int K, std::vector<std::vector<Mat> >& blkKmats,
    std::vector<PetscInt>& partX, std::vector<Mat>& Khat1Dmats) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  int nx = partX[rank];
  Khat1Dmats.resize(K, NULL);
  for(int i = 0; i < K; ++i) {
    Khat1Ddata* hatData = new Khat1Ddata; 
    MatGetVecs(blkKmats[0][0], &(hatData->u), &(hatData->uPrime));
    MatGetVecs(blkKmats[0][0], PETSC_NULL, &(hatData->tmpOut));
    hatData->blkKmats = &blkKmats;
    hatData->partX = &partX;
    hatData->numDofs = i + 1;
    hatData->K = K;
    MatCreateShell(comm, (nx*(i + 1)), (nx*(i + 1)), PETSC_DETERMINE, PETSC_DETERMINE, hatData, &(Khat1Dmats[i]));
    MatShellSetOperation(Khat1Dmats[i], MATOP_MULT, (void(*)(void))(&Khat1Dmult));
  }//end i
}

void createAll1DmatShells(int K, std::vector<MPI_Comm>& activeComms, 
    std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<Mat> >& Khat1Dmats) {
  int nlevels = activeComms.size();
  Khat1Dmats.resize(nlevels - 1);
  for(int lev = 1; lev < nlevels; ++lev) {
    if(activeComms[lev] != MPI_COMM_NULL) {
      create1DmatShells(activeComms[lev], K, blkKmats[lev - 1], partX[lev], Khat1Dmats[lev - 1]); 
    }
  }//end lev
}

void createAll1DhatPc(std::vector<std::vector<PetscInt> >& partX,
    std::vector<std::vector<std::vector<Mat> > >& blkKmats,
    std::vector<std::vector<Mat> >& Khat1Dmats, std::vector<std::vector<PC> >& hatPc) {
  hatPc.resize(Khat1Dmats.size());
  for(int i = 0; i < (Khat1Dmats.size()); ++i) {
    hatPc[i].resize(Khat1Dmats[i].size());
    for(int j = 0; j < (Khat1Dmats[i].size()); ++j) {
      MPI_Comm comm;
      PetscObjectGetComm((PetscObject)(Khat1Dmats[i][j]), &comm);
      PCCreate(comm, &(hatPc[i][j]));
      PCSetType(hatPc[i][j], PCSHELL);
      PCFD1Ddata* data = new PCFD1Ddata; 
      PCShellSetContext(hatPc[i][j], data);
      PCShellSetName(hatPc[i][j], "MyPCFD");
      PCShellSetApply(hatPc[i][j], &applyPCFD1D);
      KSP ksp;
      KSPCreate(comm, &ksp);
      KSPSetType(ksp, KSPFGMRES);
      KSPSetPCSide(ksp, PC_RIGHT);
      if(j == 0) {
        PC pc;
        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCNONE);
        KSPSetOptionsPrefix(ksp, "hat0_");
      } else {
        KSPSetPC(ksp, hatPc[i][j - 1]);
        KSPSetOptionsPrefix(ksp, "hat1_");
      }
      KSPSetOperators(ksp, Khat1Dmats[i][j], Khat1Dmats[i][j], SAME_PRECONDITIONER);
      KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
      KSPSetTolerances(ksp, 1.0e-12, 1.0e-12, PETSC_DEFAULT, 2);
      KSPSetFromOptions(ksp);
      data->ksp = ksp;
      data->partX = &(partX[i + 1]);
      data->numDofs = j + 2;
      MatGetVecs(Khat1Dmats[i][j], &(data->sol), &(data->rhs));
      MatGetVecs(blkKmats[i][0][0], &(data->u), &(data->uPrime));
    }//end j
  }//end i
}

void correctBlkKmats(int dim, std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<DM>& da,
    std::vector<std::vector<PetscInt> >& partZ, std::vector<std::vector<PetscInt> >& partY,
    std::vector<std::vector<PetscInt> >& partX, std::vector<std::vector<PetscInt> >& offsets, int K) {
  int nlevels = da.size();
  for(int lev = 1; lev < nlevels; ++lev) {
    if(da[lev] != NULL) {
      if(dim == 1) {
        blkDirichletMatCorrection1D(blkKmats[lev - 1], da[lev], partX[lev], offsets[lev], K);
      } else if(dim == 2) {
        blkDirichletMatCorrection2D(blkKmats[lev - 1], da[lev], partY[lev], 
            partX[lev], offsets[lev], K);
      } else {
        blkDirichletMatCorrection3D(blkKmats[lev - 1], da[lev], partZ[lev],
            partY[lev], partX[lev], offsets[lev], K);
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

void blkDirichletMatCorrection2D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K) {
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

  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = partX.size();

  int rj = rank/px;
  int ri = rank%px;

  if(xs == 0) {
    PetscInt xi = 0;
    PetscInt bXloc = xi - xs;
    for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
      PetscInt bYloc = yi - ys;
      PetscInt bLoc = (bYloc*partX[ri]) + bXloc;
      PetscInt bnd = offsets[rank] + bLoc;
      int dx = 0;
      for(int dy = 0; dy <= K; ++dy) {
        PetscInt bd = (dy*(K + 1)) + dx;
        for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
          if((oyi < 0) || (oyi >= Ny)) {
            continue;
          }
          PetscInt pj = rj;
          PetscInt oys = ys;
          if(oyi >= (ys + ny)) {
            pj = rj + 1;
            oys = ys + ny;
          }
          if(oyi < ys) {
            pj = rj - 1;
            oys = ys - partY[pj];
          }
          PetscInt oYloc = oyi - oys;
          for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
            if((oxi < 0) || (oxi >= Nx)) {
              continue;
            }
            PetscInt pi = ri;
            PetscInt oxs = xs;
            if(oxi >= (xs + nx)) {
              pi = ri + 1;
              oxs = xs + nx;
            }
            if(oxi < xs) {
              pi = ri - 1;
              oxs = xs - partX[pi];
            }
            PetscInt oXloc = oxi - oxs;
            int pid = (pj*px) + pi;
            PetscInt oLoc = (oYloc*partX[pi]) + oXloc;
            PetscInt oth = offsets[pid] + oLoc;
            for(int od = 0; od < dofsPerNode; ++od) {
              if(od == bd) {
                if(bnd == oth) {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              } else if(od > bd) {
                MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
              } else {
                MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end od
          }//end oxi
        }//end oyi
      }//end dy 
    }//end yi
  }
  if((xs + nx) == Nx) {
    PetscInt xi = Nx - 1;
    PetscInt bXloc = xi - xs;
    for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
      PetscInt bYloc = yi - ys;
      PetscInt bLoc = (bYloc*partX[ri]) + bXloc;
      PetscInt bnd = offsets[rank] + bLoc;
      int dx = 0;
      for(int dy = 0; dy <= K; ++dy) {
        PetscInt bd = (dy*(K + 1)) + dx;
        for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
          if((oyi < 0) || (oyi >= Ny)) {
            continue;
          }
          PetscInt pj = rj;
          PetscInt oys = ys;
          if(oyi >= (ys + ny)) {
            pj = rj + 1;
            oys = ys + ny;
          }
          if(oyi < ys) {
            pj = rj - 1;
            oys = ys - partY[pj];
          }
          PetscInt oYloc = oyi - oys;
          for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
            if((oxi < 0) || (oxi >= Nx)) {
              continue;
            }
            PetscInt pi = ri;
            PetscInt oxs = xs;
            if(oxi >= (xs + nx)) {
              pi = ri + 1;
              oxs = xs + nx;
            }
            if(oxi < xs) {
              pi = ri - 1;
              oxs = xs - partX[pi];
            }
            PetscInt oXloc = oxi - oxs;
            int pid = (pj*px) + pi;
            PetscInt oLoc = (oYloc*partX[pi]) + oXloc;
            PetscInt oth = offsets[pid] + oLoc;
            for(int od = 0; od < dofsPerNode; ++od) {
              if(od == bd) {
                if(bnd == oth) {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              } else if(od > bd) {
                MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
              } else {
                MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end od
          }//end oxi
        }//end oyi
      }//end dy 
    }//end yi
  }
  if(ys == 0) {
    PetscInt yi = 0;
    PetscInt bYloc = yi - ys;
    for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
      PetscInt bXloc = xi - xs;
      PetscInt bLoc = (bYloc*partX[ri]) + bXloc;
      PetscInt bnd = offsets[rank] + bLoc;
      int dy = 0;
      for(int dx = 0; dx <= K; ++dx) {
        PetscInt bd = (dy*(K + 1)) + dx;
        for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
          if((oyi < 0) || (oyi >= Ny)) {
            continue;
          }
          PetscInt pj = rj;
          PetscInt oys = ys;
          if(oyi >= (ys + ny)) {
            pj = rj + 1;
            oys = ys + ny;
          }
          if(oyi < ys) {
            pj = rj - 1;
            oys = ys - partY[pj];
          }
          PetscInt oYloc = oyi - oys;
          for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
            if((oxi < 0) || (oxi >= Nx)) {
              continue;
            }
            PetscInt pi = ri;
            PetscInt oxs = xs;
            if(oxi >= (xs + nx)) {
              pi = ri + 1;
              oxs = xs + nx;
            }
            if(oxi < xs) {
              pi = ri - 1;
              oxs = xs - partX[pi];
            }
            PetscInt oXloc = oxi - oxs;
            int pid = (pj*px) + pi;
            PetscInt oLoc = (oYloc*partX[pi]) + oXloc;
            PetscInt oth = offsets[pid] + oLoc;
            for(int od = 0; od < dofsPerNode; ++od) {
              if(od == bd) {
                if(bnd == oth) {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              } else if(od > bd) {
                MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
              } else {
                MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end od
          }//end oxi
        }//end oyi
      }//end dx 
    }//end xi
  }
  if((ys + ny) == Ny) {
    PetscInt yi = Ny - 1;
    PetscInt bYloc = yi - ys;
    for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
      PetscInt bXloc = xi - xs;
      PetscInt bLoc = (bYloc*partX[ri]) + bXloc;
      PetscInt bnd = offsets[rank] + bLoc;
      int dy = 0;
      for(int dx = 0; dx <= K; ++dx) {
        PetscInt bd = (dy*(K + 1)) + dx;
        for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
          if((oyi < 0) || (oyi >= Ny)) {
            continue;
          }
          PetscInt pj = rj;
          PetscInt oys = ys;
          if(oyi >= (ys + ny)) {
            pj = rj + 1;
            oys = ys + ny;
          }
          if(oyi < ys) {
            pj = rj - 1;
            oys = ys - partY[pj];
          }
          PetscInt oYloc = oyi - oys;
          for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
            if((oxi < 0) || (oxi >= Nx)) {
              continue;
            }
            PetscInt pi = ri;
            PetscInt oxs = xs;
            if(oxi >= (xs + nx)) {
              pi = ri + 1;
              oxs = xs + nx;
            }
            if(oxi < xs) {
              pi = ri - 1;
              oxs = xs - partX[pi];
            }
            PetscInt oXloc = oxi - oxs;
            int pid = (pj*px) + pi;
            PetscInt oLoc = (oYloc*partX[pi]) + oXloc;
            PetscInt oth = offsets[pid] + oLoc;
            for(int od = 0; od < dofsPerNode; ++od) {
              if(od == bd) {
                if(bnd == oth) {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              } else if(od > bd) {
                MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
              } else {
                MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end od
          }//end oxi
        }//end oyi
      }//end dx 
    }//end xi
  }

  for(int i = 0; i < (blkKmat.size()); ++i) {
    for(int j = 0; j < (blkKmat[i].size()); ++j) {
      MatAssemblyBegin(blkKmat[i][j], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(blkKmat[i][j], MAT_FINAL_ASSEMBLY);
    }//end j
  }//end i
}

void blkDirichletMatCorrection1D(std::vector<std::vector<Mat> >& blkKmat, DM da,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K) {
  PetscInt dofsPerNode;
  PetscInt Nx;
  DMDAGetInfo(da, PETSC_NULL, &Nx, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt nx;
  DMDAGetCorners(da, &xs, PETSC_NULL, PETSC_NULL, &nx, PETSC_NULL, PETSC_NULL);

  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ri = rank;

  if(xs == 0) {
    PetscInt xi = 0;
    PetscInt bLoc = xi - xs;
    PetscInt bnd = offsets[rank] + bLoc;
    PetscInt bd = 0;
    for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
      if((oxi < 0) || (oxi >= Nx)) {
        continue;
      }
      PetscInt pi = ri;
      PetscInt oxs = xs;
      if(oxi >= (xs + nx)) {
        pi = ri + 1;
        oxs = xs + nx;
      }
      if(oxi < xs) {
        pi = ri - 1;
        oxs = xs - partX[pi];
      }
      PetscInt oLoc = oxi - oxs;
      PetscInt oth = offsets[pi] + oLoc;
      for(int od = 0; od < dofsPerNode; ++od) {
        if(od == bd) {
          if(bnd == oth) {
            MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
          } else {
            MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
            MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
          }
        } else if(od > bd) {
          MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
        } else {
          MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
        }
      }//end od
    }//end oxi
  }
  if((xs + nx) == Nx) {
    PetscInt xi = Nx - 1;
    PetscInt bLoc = xi - xs;
    PetscInt bnd = offsets[rank] + bLoc;
    PetscInt bd = 0;
    for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
      if((oxi < 0) || (oxi >= Nx)) {
        continue;
      }
      PetscInt pi = ri;
      PetscInt oxs = xs;
      if(oxi >= (xs + nx)) {
        pi = ri + 1;
        oxs = xs + nx;
      }
      if(oxi < xs) {
        pi = ri - 1;
        oxs = xs - partX[pi];
      }
      PetscInt oLoc = oxi - oxs;
      PetscInt oth = offsets[pi] + oLoc;
      for(int od = 0; od < dofsPerNode; ++od) {
        if(od == bd) {
          if(bnd == oth) {
            MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
          } else {
            MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
            MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
          }
        } else if(od > bd) {
          MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
        } else {
          MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
        }
      }//end od
    }//end oxi
  }

  for(int i = 0; i < (blkKmat.size()); ++i) {
    for(int j = 0; j < (blkKmat[i].size()); ++j) {
      MatAssemblyBegin(blkKmat[i][j], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(blkKmat[i][j], MAT_FINAL_ASSEMBLY);
    }//end j
  }//end i
}

void blkDirichletMatCorrection3D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partZ,
    std::vector<PetscInt>& partY, std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K) {
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

  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = partX.size();
  int py = partY.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  if(xs == 0) {
    PetscInt xi = 0;
    PetscInt bXloc = xi - xs;
    for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
      PetscInt bZloc = zi - zs;
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        PetscInt bYloc = yi - ys;
        PetscInt bLoc = (((bZloc*partY[rj]) + bYloc)*partX[ri]) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        int dx = 0;
        for(int dz = 0; dz <= K; ++dz) {
          for(int dy = 0; dy <= K; ++dy) {
            PetscInt bd = (((dz*(K + 1)) + dy)*(K + 1)) + dx;
            for(PetscInt ozi = (zi - 1); ozi <= (zi + 1); ++ozi) {
              if((ozi < 0) || (ozi >= Nz)) {
                continue;
              }
              PetscInt pk = rk;
              PetscInt ozs = zs;
              if(ozi >= (zs + nz)) {
                pk = rk + 1;
                ozs = zs + nz;
              }
              if(ozi < zs) {
                pk = rk - 1;
                ozs = zs - partZ[pk];
              }
              PetscInt oZloc = ozi - ozs;
              for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
                if((oyi < 0) || (oyi >= Ny)) {
                  continue;
                }
                PetscInt pj = rj;
                PetscInt oys = ys;
                if(oyi >= (ys + ny)) {
                  pj = rj + 1;
                  oys = ys + ny;
                }
                if(oyi < ys) {
                  pj = rj - 1;
                  oys = ys - partY[pj];
                }
                PetscInt oYloc = oyi - oys;
                for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
                  if((oxi < 0) || (oxi >= Nx)) {
                    continue;
                  }
                  PetscInt pi = ri;
                  PetscInt oxs = xs;
                  if(oxi >= (xs + nx)) {
                    pi = ri + 1;
                    oxs = xs + nx;
                  }
                  if(oxi < xs) {
                    pi = ri - 1;
                    oxs = xs - partX[pi];
                  }
                  PetscInt oXloc = oxi - oxs;
                  int pid = (((pk*py) + pj)*px) + pi;
                  PetscInt oLoc = (((oZloc*partY[pj]) + oYloc)*partX[pi]) + oXloc;
                  PetscInt oth = offsets[pid] + oLoc;
                  for(int od = 0; od < dofsPerNode; ++od) {
                    if(od == bd) {
                      if(bnd == oth) {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    } else if(od > bd) {
                      MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    }
                  }//end od
                }//end oxi
              }//end oyi
            }//end ozi
          }//end dy
        }//end dz
      }//end yi
    }//end zi
  }
  if((xs + nx) == Nx) {
    PetscInt xi = Nx - 1;
    PetscInt bXloc = xi - xs;
    for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
      PetscInt bZloc = zi - zs;
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        PetscInt bYloc = yi - ys;
        PetscInt bLoc = (((bZloc*partY[rj]) + bYloc)*partX[ri]) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        int dx = 0;
        for(int dz = 0; dz <= K; ++dz) {
          for(int dy = 0; dy <= K; ++dy) {
            PetscInt bd = (((dz*(K + 1)) + dy)*(K + 1)) + dx;
            for(PetscInt ozi = (zi - 1); ozi <= (zi + 1); ++ozi) {
              if((ozi < 0) || (ozi >= Nz)) {
                continue;
              }
              PetscInt pk = rk;
              PetscInt ozs = zs;
              if(ozi >= (zs + nz)) {
                pk = rk + 1;
                ozs = zs + nz;
              }
              if(ozi < zs) {
                pk = rk - 1;
                ozs = zs - partZ[pk];
              }
              PetscInt oZloc = ozi - ozs;
              for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
                if((oyi < 0) || (oyi >= Ny)) {
                  continue;
                }
                PetscInt pj = rj;
                PetscInt oys = ys;
                if(oyi >= (ys + ny)) {
                  pj = rj + 1;
                  oys = ys + ny;
                }
                if(oyi < ys) {
                  pj = rj - 1;
                  oys = ys - partY[pj];
                }
                PetscInt oYloc = oyi - oys;
                for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
                  if((oxi < 0) || (oxi >= Nx)) {
                    continue;
                  }
                  PetscInt pi = ri;
                  PetscInt oxs = xs;
                  if(oxi >= (xs + nx)) {
                    pi = ri + 1;
                    oxs = xs + nx;
                  }
                  if(oxi < xs) {
                    pi = ri - 1;
                    oxs = xs - partX[pi];
                  }
                  PetscInt oXloc = oxi - oxs;
                  int pid = (((pk*py) + pj)*px) + pi;
                  PetscInt oLoc = (((oZloc*partY[pj]) + oYloc)*partX[pi]) + oXloc;
                  PetscInt oth = offsets[pid] + oLoc;
                  for(int od = 0; od < dofsPerNode; ++od) {
                    if(od == bd) {
                      if(bnd == oth) {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    } else if(od > bd) {
                      MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    }
                  }//end od
                }//end oxi
              }//end oyi
            }//end ozi
          }//end dy
        }//end dz
      }//end yi
    }//end zi
  }
  if(ys == 0) {
    PetscInt yi = 0;
    PetscInt bYloc = yi - ys;
    for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
      PetscInt bZloc = zi - zs;
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        PetscInt bXloc = xi - xs;
        PetscInt bLoc = (((bZloc*partY[rj]) + bYloc)*partX[ri]) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        int dy = 0;
        for(int dz = 0; dz <= K; ++dz) {
          for(int dx = 0; dx <= K; ++dx) {
            PetscInt bd = (((dz*(K + 1)) + dy)*(K + 1)) + dx;
            for(PetscInt ozi = (zi - 1); ozi <= (zi + 1); ++ozi) {
              if((ozi < 0) || (ozi >= Nz)) {
                continue;
              }
              PetscInt pk = rk;
              PetscInt ozs = zs;
              if(ozi >= (zs + nz)) {
                pk = rk + 1;
                ozs = zs + nz;
              }
              if(ozi < zs) {
                pk = rk - 1;
                ozs = zs - partZ[pk];
              }
              PetscInt oZloc = ozi - ozs;
              for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
                if((oyi < 0) || (oyi >= Ny)) {
                  continue;
                }
                PetscInt pj = rj;
                PetscInt oys = ys;
                if(oyi >= (ys + ny)) {
                  pj = rj + 1;
                  oys = ys + ny;
                }
                if(oyi < ys) {
                  pj = rj - 1;
                  oys = ys - partY[pj];
                }
                PetscInt oYloc = oyi - oys;
                for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
                  if((oxi < 0) || (oxi >= Nx)) {
                    continue;
                  }
                  PetscInt pi = ri;
                  PetscInt oxs = xs;
                  if(oxi >= (xs + nx)) {
                    pi = ri + 1;
                    oxs = xs + nx;
                  }
                  if(oxi < xs) {
                    pi = ri - 1;
                    oxs = xs - partX[pi];
                  }
                  PetscInt oXloc = oxi - oxs;
                  int pid = (((pk*py) + pj)*px) + pi;
                  PetscInt oLoc = (((oZloc*partY[pj]) + oYloc)*partX[pi]) + oXloc;
                  PetscInt oth = offsets[pid] + oLoc;
                  for(int od = 0; od < dofsPerNode; ++od) {
                    if(od == bd) {
                      if(bnd == oth) {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    } else if(od > bd) {
                      MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    }
                  }//end od
                }//end oxi
              }//end oyi
            }//end ozi
          }//end dx
        }//end dz
      }//end xi
    }//end zi
  }
  if((ys + ny) == Ny) {
    PetscInt yi = Ny - 1;
    PetscInt bYloc = yi - ys;
    for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
      PetscInt bZloc = zi - zs;
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        PetscInt bXloc = xi - xs;
        PetscInt bLoc = (((bZloc*partY[rj]) + bYloc)*partX[ri]) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        int dy = 0;
        for(int dz = 0; dz <= K; ++dz) {
          for(int dx = 0; dx <= K; ++dx) {
            PetscInt bd = (((dz*(K + 1)) + dy)*(K + 1)) + dx;
            for(PetscInt ozi = (zi - 1); ozi <= (zi + 1); ++ozi) {
              if((ozi < 0) || (ozi >= Nz)) {
                continue;
              }
              PetscInt pk = rk;
              PetscInt ozs = zs;
              if(ozi >= (zs + nz)) {
                pk = rk + 1;
                ozs = zs + nz;
              }
              if(ozi < zs) {
                pk = rk - 1;
                ozs = zs - partZ[pk];
              }
              PetscInt oZloc = ozi - ozs;
              for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
                if((oyi < 0) || (oyi >= Ny)) {
                  continue;
                }
                PetscInt pj = rj;
                PetscInt oys = ys;
                if(oyi >= (ys + ny)) {
                  pj = rj + 1;
                  oys = ys + ny;
                }
                if(oyi < ys) {
                  pj = rj - 1;
                  oys = ys - partY[pj];
                }
                PetscInt oYloc = oyi - oys;
                for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
                  if((oxi < 0) || (oxi >= Nx)) {
                    continue;
                  }
                  PetscInt pi = ri;
                  PetscInt oxs = xs;
                  if(oxi >= (xs + nx)) {
                    pi = ri + 1;
                    oxs = xs + nx;
                  }
                  if(oxi < xs) {
                    pi = ri - 1;
                    oxs = xs - partX[pi];
                  }
                  PetscInt oXloc = oxi - oxs;
                  int pid = (((pk*py) + pj)*px) + pi;
                  PetscInt oLoc = (((oZloc*partY[pj]) + oYloc)*partX[pi]) + oXloc;
                  PetscInt oth = offsets[pid] + oLoc;
                  for(int od = 0; od < dofsPerNode; ++od) {
                    if(od == bd) {
                      if(bnd == oth) {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    } else if(od > bd) {
                      MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    }
                  }//end od
                }//end oxi
              }//end oyi
            }//end ozi
          }//end dx
        }//end dz
      }//end xi
    }//end zi
  }
  if(zs == 0) {
    PetscInt zi = 0;
    PetscInt bZloc = zi - zs;
    for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
      PetscInt bYloc = yi - ys;
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        PetscInt bXloc = xi - xs;
        PetscInt bLoc = (((bZloc*partY[rj]) + bYloc)*partX[ri]) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        int dz = 0;
        for(int dy = 0; dy <= K; ++dy) {
          for(int dx = 0; dx <= K; ++dx) {
            PetscInt bd = (((dz*(K + 1)) + dy)*(K + 1)) + dx;
            for(PetscInt ozi = (zi - 1); ozi <= (zi + 1); ++ozi) {
              if((ozi < 0) || (ozi >= Nz)) {
                continue;
              }
              PetscInt pk = rk;
              PetscInt ozs = zs;
              if(ozi >= (zs + nz)) {
                pk = rk + 1;
                ozs = zs + nz;
              }
              if(ozi < zs) {
                pk = rk - 1;
                ozs = zs - partZ[pk];
              }
              PetscInt oZloc = ozi - ozs;
              for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
                if((oyi < 0) || (oyi >= Ny)) {
                  continue;
                }
                PetscInt pj = rj;
                PetscInt oys = ys;
                if(oyi >= (ys + ny)) {
                  pj = rj + 1;
                  oys = ys + ny;
                }
                if(oyi < ys) {
                  pj = rj - 1;
                  oys = ys - partY[pj];
                }
                PetscInt oYloc = oyi - oys;
                for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
                  if((oxi < 0) || (oxi >= Nx)) {
                    continue;
                  }
                  PetscInt pi = ri;
                  PetscInt oxs = xs;
                  if(oxi >= (xs + nx)) {
                    pi = ri + 1;
                    oxs = xs + nx;
                  }
                  if(oxi < xs) {
                    pi = ri - 1;
                    oxs = xs - partX[pi];
                  }
                  PetscInt oXloc = oxi - oxs;
                  int pid = (((pk*py) + pj)*px) + pi;
                  PetscInt oLoc = (((oZloc*partY[pj]) + oYloc)*partX[pi]) + oXloc;
                  PetscInt oth = offsets[pid] + oLoc;
                  for(int od = 0; od < dofsPerNode; ++od) {
                    if(od == bd) {
                      if(bnd == oth) {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    } else if(od > bd) {
                      MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    }
                  }//end od
                }//end oxi
              }//end oyi
            }//end ozi
          }//end dx
        }//end dy
      }//end xi
    }//end yi
  }
  if((zs + nz) == Nz) {
    PetscInt zi = Nz - 1;
    PetscInt bZloc = zi - zs;
    for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
      PetscInt bYloc = yi - ys;
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        PetscInt bXloc = xi - xs;
        PetscInt bLoc = (((bZloc*partY[rj]) + bYloc)*partX[ri]) + bXloc;
        PetscInt bnd = offsets[rank] + bLoc;
        int dz = 0;
        for(int dy = 0; dy <= K; ++dy) {
          for(int dx = 0; dx <= K; ++dx) {
            PetscInt bd = (((dz*(K + 1)) + dy)*(K + 1)) + dx;
            for(PetscInt ozi = (zi - 1); ozi <= (zi + 1); ++ozi) {
              if((ozi < 0) || (ozi >= Nz)) {
                continue;
              }
              PetscInt pk = rk;
              PetscInt ozs = zs;
              if(ozi >= (zs + nz)) {
                pk = rk + 1;
                ozs = zs + nz;
              }
              if(ozi < zs) {
                pk = rk - 1;
                ozs = zs - partZ[pk];
              }
              PetscInt oZloc = ozi - ozs;
              for(PetscInt oyi = (yi - 1); oyi <= (yi + 1); ++oyi) {
                if((oyi < 0) || (oyi >= Ny)) {
                  continue;
                }
                PetscInt pj = rj;
                PetscInt oys = ys;
                if(oyi >= (ys + ny)) {
                  pj = rj + 1;
                  oys = ys + ny;
                }
                if(oyi < ys) {
                  pj = rj - 1;
                  oys = ys - partY[pj];
                }
                PetscInt oYloc = oyi - oys;
                for(PetscInt oxi = (xi - 1); oxi <= (xi + 1); ++oxi) {
                  if((oxi < 0) || (oxi >= Nx)) {
                    continue;
                  }
                  PetscInt pi = ri;
                  PetscInt oxs = xs;
                  if(oxi >= (xs + nx)) {
                    pi = ri + 1;
                    oxs = xs + nx;
                  }
                  if(oxi < xs) {
                    pi = ri - 1;
                    oxs = xs - partX[pi];
                  }
                  PetscInt oXloc = oxi - oxs;
                  int pid = (((pk*py) + pj)*px) + pi;
                  PetscInt oLoc = (((oZloc*partY[pj]) + oYloc)*partX[pi]) + oXloc;
                  PetscInt oth = offsets[pid] + oLoc;
                  for(int od = 0; od < dofsPerNode; ++od) {
                    if(od == bd) {
                      if(bnd == oth) {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValues(blkKmat[bd][0], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValues(blkKmat[bd][0], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    } else if(od > bd) {
                      MatSetValues(blkKmat[bd][od - bd], 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                    } else {
                      MatSetValues(blkKmat[od][bd - od], 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                    }
                  }//end od
                }//end oxi
              }//end oyi
            }//end ozi
          }//end dx
        }//end dy
      }//end xi
    }//end yi
  }

  for(int i = 0; i < (blkKmat.size()); ++i) {
    for(int j = 0; j < (blkKmat[i].size()); ++j) {
      MatAssemblyBegin(blkKmat[i][j], MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(blkKmat[i][j], MAT_FINAL_ASSEMBLY);
    }//end j
  }//end i
}

void destroyPCFD1Ddata(PCFD1Ddata* data) {
  KSPDestroy(&(data->ksp));
  VecDestroy(&(data->rhs));
  VecDestroy(&(data->sol));
  VecDestroy(&(data->u));
  VecDestroy(&(data->uPrime));
  delete data;
}

void destroyKhat1Ddata(Khat1Ddata* data) {
  VecDestroy(&(data->u));
  VecDestroy(&(data->uPrime));
  VecDestroy(&(data->tmpOut));
  delete data;
}


