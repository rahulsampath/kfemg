
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

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
  long double hy;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz;
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



