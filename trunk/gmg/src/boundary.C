
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

void dirichletMatrixCorrectionBlkUpper(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets) {
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

void dirichletMatrixCorrectionBlkDiag(Mat Kblk, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets) {
  PetscInt dim;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DMDAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

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

void dirichletMatrixCorrection(Mat Kmat, DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, 
    std::vector<PetscInt>& lx, std::vector<PetscInt>& offsets) {
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

void zeroBoundaries(DM da, Vec vec) {
  PetscInt dim;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  DMDAGetInfo(da, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  if(dim == 1) {
    PetscScalar** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      arr[0][0] = 0.0;
    }
    if((xs + nx) == Nx) {
      arr[Nx - 1][0] = 0.0;
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
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
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else {
    PetscScalar**** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
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
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  }
}



