
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void correctBlkKmats(int dim, std::vector<std::vector<std::vector<Mat> > >& blkKmats, std::vector<DM>& da,
    std::vector<std::vector<PetscInt> >& partY, std::vector<std::vector<PetscInt> >& partX, 
    std::vector<std::vector<PetscInt> >& offsets, int K) {
  for(int lev = 0; lev < (da.size()); ++lev) {
    if(da[lev] != NULL) {
      if(dim == 1) {
        blkDirichletMatCorrection1D(blkKmats[lev], da[lev], offsets[lev], K);
      } else if(dim == 2) {
        blkDirichletMatCorrection2D(blkKmats[lev], da[lev], partX[lev], offsets[lev], K);
      } else {
        blkDirichletMatCorrection3D(blkKmats[lev], da[lev], partY[lev], partX[lev], offsets[lev], K);
      }
    }
  }//end lev
}

void correctKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, int K) {
  for(int lev = 0; lev < (Kmat.size()); ++lev) {
    if(Kmat[lev] != NULL) {
      dirichletMatrixCorrection(Kmat[lev], da[lev], K);
    }
  }//end lev
}

void setBoundaries(DM da, Vec vec, const int K) {
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

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  long double hy = 0;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz = 0;
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  }

  if(dim == 1) {
    PetscScalar** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      arr[0][0] = solution1D(0);
    }
    if((xs + nx) == Nx) {
      arr[Nx - 1][0] = solution1D(1);
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        long double y = ((static_cast<long double>(yi)) * hy);
        for(int d = 0; d <= K; ++d) {
          arr[yi][0][d*(K + 1)] = myIntPow((0.5L * hy), d) * solutionDerivative2D(0, y, 0, d);
        }//end d
      }//end yi
    }
    if((xs + nx) == Nx) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        long double y = ((static_cast<long double>(yi)) * hy);
        for(int d = 0; d <= K; ++d) {
          arr[yi][Nx - 1][d*(K + 1)] = myIntPow((0.5L * hy), d) * solutionDerivative2D(1, y, 0, d);
        }//end d
      }//end yi
    }
    if(ys == 0) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        long double x = ((static_cast<long double>(xi)) * hx);
        for(int d = 0; d <= K; ++d) {
          arr[0][xi][d] = myIntPow((0.5L * hx), d) * solutionDerivative2D(x, 0, d, 0);
        }//end d
      }//end xi
    }
    if((ys + ny) == Ny) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        long double x = ((static_cast<long double>(xi)) * hx);
        for(int d = 0; d <= K; ++d) {
          arr[Ny - 1][xi][d] = myIntPow((0.5L * hx), d) * solutionDerivative2D(x, 1, d, 0);
        }//end d
      }//end xi
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else {
    PetscScalar**** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        long double z = ((static_cast<long double>(zi)) * hz);
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          long double y = ((static_cast<long double>(yi)) * hy);
          for(int dz = 0; dz <= K; ++dz) {
            for(int dy = 0; dy <= K; ++dy) {
              int dof = ((dz*(K + 1)) + dy)*(K + 1);
              arr[zi][yi][0][dof] = myIntPow((0.5L * hz), dz) * myIntPow((0.5L * hy), dy) 
                * solutionDerivative3D(0, y, z, 0, dy, dz);
            }//end dy
          }//end dz
        }//end yi
      }//end zi
    }
    if((xs + nx) == Nx) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        long double z = ((static_cast<long double>(zi)) * hz);
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          long double y = ((static_cast<long double>(yi)) * hy);
          for(int dz = 0; dz <= K; ++dz) {
            for(int dy = 0; dy <= K; ++dy) {
              int dof = ((dz*(K + 1)) + dy)*(K + 1);
              arr[zi][yi][Nx - 1][dof] = myIntPow((0.5L * hz), dz) * myIntPow((0.5L * hy), dy) 
                * solutionDerivative3D(1, y, z, 0, dy, dz);
            }//end dy
          }//end dz
        }//end yi
      }//end zi
    }
    if(ys == 0) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        long double z = ((static_cast<long double>(zi)) * hz);
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          long double x = ((static_cast<long double>(xi)) * hx);
          for(int dz = 0; dz <= K; ++dz) {
            for(int dx = 0; dx <= K; ++dx) {
              int dof = (dz*(K + 1)*(K + 1)) + dx;
              arr[zi][0][xi][dof] = myIntPow((0.5L * hz), dz) * myIntPow((0.5L * hx), dx) 
                * solutionDerivative3D(x, 0, z, dx, 0, dz);
            }//end dx
          }//end dz
        }//end xi
      }//end zi
    }
    if((ys + ny) == Ny) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        long double z = ((static_cast<long double>(zi)) * hz);
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          long double x = ((static_cast<long double>(xi)) * hx);
          for(int dz = 0; dz <= K; ++dz) {
            for(int dx = 0; dx <= K; ++dx) {
              int dof = (dz*(K + 1)*(K + 1)) + dx;
              arr[zi][Ny - 1][xi][dof] = myIntPow((0.5L * hz), dz) * myIntPow((0.5L * hx), dx) 
                * solutionDerivative3D(x, 1, z, dx, 0, dz);
            }//end dx
          }//end dz
        }//end xi
      }//end zi
    }
    if(zs == 0) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        long double y = ((static_cast<long double>(yi)) * hy);
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          long double x = ((static_cast<long double>(xi)) * hx);
          for(int dy = 0; dy <= K; ++dy) {
            for(int dx = 0; dx <= K; ++dx) {
              int dof = (dy*(K + 1)) + dx;
              arr[0][yi][xi][dof] = myIntPow((0.5L * hy), dy) * myIntPow((0.5L * hx), dx) 
                * solutionDerivative3D(x, y, 0, dx, dy, 0);
            }//end dx
          }//end dy
        }//end xi
      }//end yi
    }
    if((zs + nz) == Nz) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        long double y = ((static_cast<long double>(yi)) * hy);
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          long double x = ((static_cast<long double>(xi)) * hx);
          for(int dy = 0; dy <= K; ++dy) {
            for(int dx = 0; dx <= K; ++dx) {
              int dof = (dy*(K + 1)) + dx;
              arr[Nz - 1][yi][xi][dof] = myIntPow((0.5L * hy), dy) * myIntPow((0.5L * hx), dx) 
                * solutionDerivative3D(x, y, 1, dx, dy, 0);
            }//end dx
          }//end dy
        }//end xi
      }//end yi
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  }
}

void dirichletMatrixCorrection(Mat Kmat, DM da, const int K) {
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

  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;

  if(dim == 1) {
    if(xs == 0) {
      MatStencil bnd;
      bnd.i = 0;
      bnd.c = 0;
      for(int node = 0; node < 2; ++node) {
        for(int dof = 0; dof < dofsPerNode; ++dof) {
          MatStencil oth;
          oth.i = node;
          oth.c = dof;
          if((bnd.i == oth.i) && (bnd.c == oth.c)) {
            MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
          } else {
            MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
            MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
          }
        }//end dof
      }//end node
    }
    if((xs + nx) == Nx) {
      MatStencil bnd;
      bnd.i = Nx - 1;
      bnd.c = 0;
      for(int node = 1; node <= 2; ++node) {
        for(int dof = 0; dof < dofsPerNode; ++dof) {
          MatStencil oth;
          oth.i = Nx - node;
          oth.c = dof;
          if((bnd.i == oth.i) && (bnd.c == oth.c)) {
            MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
          } else {
            MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
            MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
          }
        }//end dof
      }//end node
    }
  } else if(dim == 2) {
    if(xs == 0) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        MatStencil bnd;
        bnd.j = yi;
        bnd.i = 0;
        for(int d = 0; d <= K; ++d) {
          bnd.c = d*(K + 1);
          for(int nodeY = -1; nodeY < 2; ++nodeY) {
            if((yi == 0) && (nodeY == -1)) {
              continue;
            }
            if((yi == (Ny - 1)) && (nodeY == 1)) {
              continue;
            }
            for(int nodeX = 0; nodeX < 2; ++nodeX) {
              for(int dof = 0; dof < dofsPerNode; ++dof) {
                MatStencil oth;
                oth.j = yi + nodeY;
                oth.i = nodeX;
                oth.c = dof;
                if((bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              }//end dof
            }//end nodeX
          }//end nodeY
        }//end d
      }//end yi
    }
    if((xs + nx) == Nx) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        MatStencil bnd;
        bnd.j = yi;
        bnd.i = Nx - 1;
        for(int d = 0; d <= K; ++d) {
          bnd.c = d*(K + 1);
          for(int nodeY = -1; nodeY < 2; ++nodeY) {
            if((yi == 0) && (nodeY == -1)) {
              continue;
            }
            if((yi == (Ny - 1)) && (nodeY == 1)) {
              continue;
            }
            for(int nodeX = -1; nodeX < 1; ++nodeX) {
              for(int dof = 0; dof < dofsPerNode; ++dof) {
                MatStencil oth;
                oth.j = yi + nodeY;
                oth.i = Nx - 1 + nodeX;
                oth.c = dof;
                if((bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              }//end dof
            }//end nodeX
          }//end nodeY
        }//end d
      }//end yi
    }
    if(ys == 0) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        MatStencil bnd;
        bnd.j = 0;
        bnd.i = xi;
        for(int d = 0; d <= K; ++d) {
          bnd.c = d;
          for(int nodeY = 0; nodeY < 2; ++nodeY) {
            for(int nodeX = -1; nodeX < 2; ++nodeX) {
              if((xi == 0) && (nodeX == -1)) {
                continue;
              }
              if((xi == (Nx - 1)) && (nodeX == 1)) {
                continue;
              }
              for(int dof = 0; dof < dofsPerNode; ++dof) {
                MatStencil oth;
                oth.j = nodeY;
                oth.i = xi + nodeX;
                oth.c = dof;
                if((bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              }//end dof
            }//end nodeX
          }//end nodeY
        }//end d
      }//end xi
    }
    if((ys + ny) == Ny) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        MatStencil bnd;
        bnd.j = Ny - 1;
        bnd.i = xi;
        for(int d = 0; d <= K; ++d) {
          bnd.c = d;
          for(int nodeY = -1; nodeY < 1; ++nodeY) {
            for(int nodeX = -1; nodeX < 2; ++nodeX) {
              if((xi == 0) && (nodeX == -1)) {
                continue;
              }
              if((xi == (Nx - 1)) && (nodeX == 1)) {
                continue;
              }
              for(int dof = 0; dof < dofsPerNode; ++dof) {
                MatStencil oth;
                oth.j = Ny - 1 + nodeY;
                oth.i = xi + nodeX;
                oth.c = dof;
                if((bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                } else {
                  MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                  MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                }
              }//end dof
            }//end nodeX
          }//end nodeY
        }//end d
      }//end xi
    }
  } else {
    if(xs == 0) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          MatStencil bnd;
          bnd.k = zi;
          bnd.j = yi;
          bnd.i = 0;
          for(int dz = 0; dz <= K; ++dz) {
            for(int dy = 0; dy <= K; ++dy) {
              bnd.c = ((dz*(K + 1)) + dy)*(K + 1);
              for(int nodeZ = -1; nodeZ < 2; ++nodeZ) {
                if((bnd.k == 0) && (nodeZ == -1)) {
                  continue;
                }
                if((bnd.k == (Nz - 1)) && (nodeZ == 1)) {
                  continue;
                }
                for(int nodeY = -1; nodeY < 2; ++nodeY) {
                  if((bnd.j == 0) && (nodeY == -1)) {
                    continue;
                  }
                  if((bnd.j == (Ny - 1)) && (nodeY == 1)) {
                    continue;
                  }
                  for(int nodeX = -1; nodeX < 2; ++nodeX) {
                    if((bnd.i == 0) && (nodeX == -1)) {
                      continue;
                    }
                    if((bnd.i == (Nx - 1)) && (nodeX == 1)) {
                      continue;
                    }
                    for(int dof = 0; dof < dofsPerNode; ++dof) {
                      MatStencil oth;
                      oth.k = bnd.k + nodeZ;
                      oth.j = bnd.j + nodeY;
                      oth.i = bnd.i + nodeX;
                      oth.c = dof;
                      if((bnd.k == oth.k) && (bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    }//end dof
                  }//end nodeX
                }//end nodeY
              }//end nodeZ
            }//end dy
          }//end dz
        }//end yi
      }//end zi
    }
    if((xs + nx) == Nx) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          MatStencil bnd;
          bnd.k = zi;
          bnd.j = yi;
          bnd.i = Nx - 1;
          for(int dz = 0; dz <= K; ++dz) {
            for(int dy = 0; dy <= K; ++dy) {
              bnd.c = ((dz*(K + 1)) + dy)*(K + 1);
              for(int nodeZ = -1; nodeZ < 2; ++nodeZ) {
                if((bnd.k == 0) && (nodeZ == -1)) {
                  continue;
                }
                if((bnd.k == (Nz - 1)) && (nodeZ == 1)) {
                  continue;
                }
                for(int nodeY = -1; nodeY < 2; ++nodeY) {
                  if((bnd.j == 0) && (nodeY == -1)) {
                    continue;
                  }
                  if((bnd.j == (Ny - 1)) && (nodeY == 1)) {
                    continue;
                  }
                  for(int nodeX = -1; nodeX < 2; ++nodeX) {
                    if((bnd.i == 0) && (nodeX == -1)) {
                      continue;
                    }
                    if((bnd.i == (Nx - 1)) && (nodeX == 1)) {
                      continue;
                    }
                    for(int dof = 0; dof < dofsPerNode; ++dof) {
                      MatStencil oth;
                      oth.k = bnd.k + nodeZ;
                      oth.j = bnd.j + nodeY;
                      oth.i = bnd.i + nodeX;
                      oth.c = dof;
                      if((bnd.k == oth.k) && (bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    }//end dof
                  }//end nodeX
                }//end nodeY
              }//end nodeZ
            }//end dy
          }//end dz
        }//end yi
      }//end zi
    }
    if(ys == 0) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          MatStencil bnd;
          bnd.k = zi;
          bnd.j = 0;
          bnd.i = xi;
          for(int dz = 0; dz <= K; ++dz) {
            for(int dx = 0; dx <= K; ++dx) {
              bnd.c = (dz*(K + 1)*(K + 1)) + dx;
              for(int nodeZ = -1; nodeZ < 2; ++nodeZ) {
                if((bnd.k == 0) && (nodeZ == -1)) {
                  continue;
                }
                if((bnd.k == (Nz - 1)) && (nodeZ == 1)) {
                  continue;
                }
                for(int nodeY = -1; nodeY < 2; ++nodeY) {
                  if((bnd.j == 0) && (nodeY == -1)) {
                    continue;
                  }
                  if((bnd.j == (Ny - 1)) && (nodeY == 1)) {
                    continue;
                  }
                  for(int nodeX = -1; nodeX < 2; ++nodeX) {
                    if((bnd.i == 0) && (nodeX == -1)) {
                      continue;
                    }
                    if((bnd.i == (Nx - 1)) && (nodeX == 1)) {
                      continue;
                    }
                    for(int dof = 0; dof < dofsPerNode; ++dof) {
                      MatStencil oth;
                      oth.k = bnd.k + nodeZ;
                      oth.j = bnd.j + nodeY;
                      oth.i = bnd.i + nodeX;
                      oth.c = dof;
                      if((bnd.k == oth.k) && (bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    }//end dof
                  }//end nodeX
                }//end nodeY
              }//end nodeZ
            }//end dx
          }//end dz
        }//end xi
      }//end zi
    }
    if((ys + ny) == Ny) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          MatStencil bnd;
          bnd.k = zi;
          bnd.j = Ny - 1;
          bnd.i = xi;
          for(int dz = 0; dz <= K; ++dz) {
            for(int dx = 0; dx <= K; ++dx) {
              bnd.c = (dz*(K + 1)*(K + 1)) + dx;
              for(int nodeZ = -1; nodeZ < 2; ++nodeZ) {
                if((bnd.k == 0) && (nodeZ == -1)) {
                  continue;
                }
                if((bnd.k == (Nz - 1)) && (nodeZ == 1)) {
                  continue;
                }
                for(int nodeY = -1; nodeY < 2; ++nodeY) {
                  if((bnd.j == 0) && (nodeY == -1)) {
                    continue;
                  }
                  if((bnd.j == (Ny - 1)) && (nodeY == 1)) {
                    continue;
                  }
                  for(int nodeX = -1; nodeX < 2; ++nodeX) {
                    if((bnd.i == 0) && (nodeX == -1)) {
                      continue;
                    }
                    if((bnd.i == (Nx - 1)) && (nodeX == 1)) {
                      continue;
                    }
                    for(int dof = 0; dof < dofsPerNode; ++dof) {
                      MatStencil oth;
                      oth.k = bnd.k + nodeZ;
                      oth.j = bnd.j + nodeY;
                      oth.i = bnd.i + nodeX;
                      oth.c = dof;
                      if((bnd.k == oth.k) && (bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    }//end dof
                  }//end nodeX
                }//end nodeY
              }//end nodeZ
            }//end dx
          }//end dz
        }//end xi
      }//end zi
    }
    if(zs == 0) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          MatStencil bnd;
          bnd.k = 0;
          bnd.j = yi;
          bnd.i = xi;
          for(int dy = 0; dy <= K; ++dy) {
            for(int dx = 0; dx <= K; ++dx) {
              bnd.c = (dy*(K + 1)) + dx;
              for(int nodeZ = -1; nodeZ < 2; ++nodeZ) {
                if((bnd.k == 0) && (nodeZ == -1)) {
                  continue;
                }
                if((bnd.k == (Nz - 1)) && (nodeZ == 1)) {
                  continue;
                }
                for(int nodeY = -1; nodeY < 2; ++nodeY) {
                  if((bnd.j == 0) && (nodeY == -1)) {
                    continue;
                  }
                  if((bnd.j == (Ny - 1)) && (nodeY == 1)) {
                    continue;
                  }
                  for(int nodeX = -1; nodeX < 2; ++nodeX) {
                    if((bnd.i == 0) && (nodeX == -1)) {
                      continue;
                    }
                    if((bnd.i == (Nx - 1)) && (nodeX == 1)) {
                      continue;
                    }
                    for(int dof = 0; dof < dofsPerNode; ++dof) {
                      MatStencil oth;
                      oth.k = bnd.k + nodeZ;
                      oth.j = bnd.j + nodeY;
                      oth.i = bnd.i + nodeX;
                      oth.c = dof;
                      if((bnd.k == oth.k) && (bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    }//end dof
                  }//end nodeX
                }//end nodeY
              }//end nodeZ
            }//end dx
          }//end dy
        }//end xi
      }//end yi
    }
    if((zs + nz) == Nz) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          MatStencil bnd;
          bnd.k = Nz - 1;
          bnd.j = yi;
          bnd.i = xi;
          for(int dy = 0; dy <= K; ++dy) {
            for(int dx = 0; dx <= K; ++dx) {
              bnd.c = (dy*(K + 1)) + dx;
              for(int nodeZ = -1; nodeZ < 2; ++nodeZ) {
                if((bnd.k == 0) && (nodeZ == -1)) {
                  continue;
                }
                if((bnd.k == (Nz - 1)) && (nodeZ == 1)) {
                  continue;
                }
                for(int nodeY = -1; nodeY < 2; ++nodeY) {
                  if((bnd.j == 0) && (nodeY == -1)) {
                    continue;
                  }
                  if((bnd.j == (Ny - 1)) && (nodeY == 1)) {
                    continue;
                  }
                  for(int nodeX = -1; nodeX < 2; ++nodeX) {
                    if((bnd.i == 0) && (nodeX == -1)) {
                      continue;
                    }
                    if((bnd.i == (Nx - 1)) && (nodeX == 1)) {
                      continue;
                    }
                    for(int dof = 0; dof < dofsPerNode; ++dof) {
                      MatStencil oth;
                      oth.k = bnd.k + nodeZ;
                      oth.j = bnd.j + nodeY;
                      oth.i = bnd.i + nodeX;
                      oth.c = dof;
                      if((bnd.k == oth.k) && (bnd.j == oth.j) && (bnd.i == oth.i) && (bnd.c == oth.c)) {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
                      } else {
                        MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                        MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
                      }
                    }//end dof
                  }//end nodeX
                }//end nodeY
              }//end nodeZ
            }//end dx
          }//end dy
        }//end xi
      }//end yi
    }
  }

  MatAssemblyBegin(Kmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FINAL_ASSEMBLY);
}

void blkDirichletMatCorrection3D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partY,
    std::vector<PetscInt>& partX, std::vector<PetscInt>& offsets, int K) {
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

void blkDirichletMatCorrection2D(std::vector<std::vector<Mat> >& blkKmat, DM da, std::vector<PetscInt>& partX,
    std::vector<PetscInt>& offsets, int K) {
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
    std::vector<PetscInt>& offsets, int K) {
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



