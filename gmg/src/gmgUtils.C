
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"
#include "petscmg.h"

void buildPmat(std::vector<Mat>& Pmat, std::vector<Vec>& tmpCvec, std::vector<DA>& da,
    std::vector<MPI_Comm>& activeComms, std::vector<int>& activeNpes, int dim, int dofsPerNode,
    std::vector<long long int>& coeffs, const unsigned int K) {
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
      computePmat(Pmat[lev], da[lev], da[lev + 1], coeffs, K);
    }
  }//end lev
}

void computePmat(Mat Pmat, DA dac, DA daf, std::vector<long long int>& coeffs, const unsigned int K) {
}

void buildKmat(std::vector<Mat>& Kmat, std::vector<DA>& da, std::vector<long long int>& coeffs, const unsigned int K) {
  Kmat.resize(da.size(), NULL);
  for(int i = 0; i < (da.size()); ++i) {
    if(da[i] != NULL) {
      DAGetMatrix(da[i], MATAIJ, &(Kmat[i]));
      computeKmat(Kmat[i], da[i], coeffs, K);
      dirichletMatrixCorrection(Kmat[i], da[i]);
    }
  }//end i
}

void computeKmat(Mat Kmat, DA da, std::vector<long long int>& coeffs, const unsigned int K) {
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
    createPoisson1DelementMatrix(K, coeffs, hx, elemMat);
  } else if(dim == 2) {
    createPoisson2DelementMatrix(K, coeffs, hy, hx, elemMat);
  } else {
    createPoisson3DelementMatrix(K, coeffs, hz, hy, hx, elemMat);
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
  VecSetRandom(tmpSol, rndCtx);
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

void createKSP(std::vector<KSP>& ksp, std::vector<Mat>& Kmat, std::vector<MPI_Comm>& activeComms) {
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
        KSPRichardsonSetScale(ksp[lev], 1.0);
        KSPSetPreconditionerSide(ksp[lev], PC_LEFT);
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
    int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx, MPI_Comm globalComm) {
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

  MPI_Group globalGroup;
  MPI_Comm_group(globalComm, &globalGroup);

  int* rankList = new int[globalNpes];
  for(int i = 0; i < globalNpes; ++i) {
    rankList[i] = i;
  }//end for i

  //0 is the coarsest level.
  for(int lev = 0; lev < numLevels; ++lev) {
    int px, py, pz;
    int maxNpes;
    if(lev == 0) {
      maxNpes = maxCoarseNpes;
    } else {
      maxNpes = globalNpes;
    }
    computePartition(dim, Nz[lev], Ny[lev], Nx[lev], maxNpes, pz, py, px);
    activeNpes[lev] = (px*py*pz);
    std::cout<<"Active Npes for Level "<<lev<<" = "<<(activeNpes[lev])<<std::endl;
    if(lev > 0) {
      assert(activeNpes[lev] >= activeNpes[lev - 1]);
    }
    if(globalRank < (activeNpes[lev])) {
      MPI_Group subGroup;
      MPI_Group_incl(globalGroup, (activeNpes[lev]), rankList, &subGroup);
      MPI_Comm_create(globalComm, subGroup, &(activeComms[lev]));
      MPI_Group_free(&subGroup);
      DACreate(activeComms[lev], dim, DA_NONPERIODIC, DA_STENCIL_BOX, (Nx[lev]), (Ny[lev]), (Nz[lev]),
          px, py, pz, dofsPerNode, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, (&(da[lev])));
    } else {
      MPI_Comm_create(globalComm, MPI_GROUP_EMPTY, &(activeComms[lev]));
      assert(activeComms[lev] == MPI_COMM_NULL);
      da[lev] = NULL;
    }
  }//end lev

  delete [] rankList;
  MPI_Group_free(&globalGroup);
}

void computePartition(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int maxNpes, int &pz, int &py, int &px) {
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

  for(int d = 0; d < 3; ++d) {
    if(Nx == Nlist[d]) {
      px = pList[d];
      Nlist.erase(Nlist.begin() + d);
      pList.erase(pList.begin() + d);
      break;
    }
  }//end d

  for(int d = 0; d < 2; ++d) {
    if(Ny == Nlist[d]) {
      py = pList[d];
      Nlist.erase(Nlist.begin() + d);
      pList.erase(pList.begin() + d);
      break;
    }
  }//end d

  assert(Nz == Nlist[0]);
  pz = pList[0];

  assert((px*py*pz) <= maxNpes);
  assert(px >= 1);
  assert(py >= 1);
  assert(pz >= 1);
  assert(px <= Nx);
  assert(py <= Ny);
  assert(pz <= Nz);
}

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx) {
  PetscInt currNx = 17;
  PetscInt currNy = 1;
  PetscInt currNz = 1;

  assert(dim > 0);
  assert(dim <= 3);

  PetscOptionsGetInt(PETSC_NULL, "-finestNx", &currNx, PETSC_NULL);
  std::cout<<"Nx (Finest) = "<<currNx<<std::endl;
  if(dim > 1) {
    PetscOptionsGetInt(PETSC_NULL, "-finestNy", &currNy, PETSC_NULL);
    std::cout<<"Ny (Finest) = "<<currNy<<std::endl;
  }
  if(dim > 2) {
    PetscOptionsGetInt(PETSC_NULL, "-finestNz", &currNz, PETSC_NULL);
    std::cout<<"Nz (Finest) = "<<currNz<<std::endl;
  }

  PetscInt maxNumLevels = 20;
  PetscOptionsGetInt(PETSC_NULL, "-maxNumLevels", &maxNumLevels, PETSC_NULL);
  std::cout<<"MaxNumLevels = "<<maxNumLevels<<std::endl;

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
    if( (currNx < 3) || ((currNx%2) == 0) ) {
      break;
    }
    currNx = 1 + ((currNx - 1)/2); 
    if(dim > 1) {
      if( (currNy < 3) || ((currNy%2) == 0) ) {
        break;
      }
      currNy = 1 + ((currNy - 1)/2); 
    }
    if(dim > 2) {
      if( (currNz < 3) || ((currNz%2) == 0) ) {
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

  std::cout<<"ActualNumLevels = "<<(Nx.size())<<std::endl;
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


