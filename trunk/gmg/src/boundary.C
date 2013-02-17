
#include "gmg/include/boundary.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void correctKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, int K) {
  int nlevels = da.size();
  for(int lev = 0; lev < nlevels; ++lev) {
    if(da[lev] != NULL) {
      dirichletMatrixCorrection(Kmat[lev], da[lev], K);
    }
  }//end lev
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


