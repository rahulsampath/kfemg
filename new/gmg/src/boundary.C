
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#include <cassert>

void dirichletMatrixCorrection(Mat Kmat, DM da) {
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
          if((node == 0) && (dof == 0)) {
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
          if((node == 1) && (dof == 0)) {
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
        bnd.c = 0;
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
              if((nodeY == 0) && (nodeX == 0) && (dof == 0)) {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
              } else {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end dof
          }//end nodeX
        }//end nodeY
      }//end yi
    }
    if((xs + nx) == Nx) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        MatStencil bnd;
        bnd.j = yi;
        bnd.i = Nx - 1;
        bnd.c = 0;
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
              if((nodeY == 0) && (nodeX == 0) && (dof == 0)) {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
              } else {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end dof
          }//end nodeX
        }//end nodeY
      }//end yi
    }
    if(ys == 0) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        MatStencil bnd;
        bnd.j = 0;
        bnd.i = xi;
        bnd.c = 0;
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
              if((nodeY == 0) && (nodeX == 0) && (dof == 0)) {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
              } else {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end dof
          }//end nodeX
        }//end nodeY
      }//end xi
    }
    if((ys + ny) == Ny) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        MatStencil bnd;
        bnd.j = Ny - 1;
        bnd.i = xi;
        bnd.c = 0;
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
              if((nodeY == 0) && (nodeX == 0) && (dof == 0)) {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &bnd, &one, INSERT_VALUES);
              } else {
                MatSetValuesStencil(Kmat, 1, &bnd, 1, &oth, &zero, INSERT_VALUES);
                MatSetValuesStencil(Kmat, 1, &oth, 1, &bnd, &zero, INSERT_VALUES);
              }
            }//end dof
          }//end nodeX
        }//end nodeY
      }//end xi
    }
  } else {
    assert(false);
  }

  MatAssemblyBegin(Kmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FINAL_ASSEMBLY);
}

void setBoundaries(DM da, Vec vec) {
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

  const int solXfac = 1;
  const int solYfac = 1;

  if(dim == 1) {
    PetscScalar** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      arr[0][0] = solution1D(0, solXfac);
    }
    if((xs + nx) == Nx) {
      arr[Nx - 1][0] = solution1D(1, solXfac);
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        arr[yi][0][0] = solution2D(0, ((static_cast<long double>(yi)) * hy), solXfac, solYfac);
      }//end yi
    }
    if((xs + nx) == Nx) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        arr[yi][Nx - 1][0] = solution2D(1, ((static_cast<long double>(yi)) * hy), solXfac, solYfac);
      }//end yi
    }
    if(ys == 0) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        arr[0][xi][0] = solution2D(((static_cast<long double>(xi)) * hx), 0, solXfac, solYfac);
      }//end xi
    }
    if((ys + ny) == Ny) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        arr[Ny - 1][xi][0] = solution2D(((static_cast<long double>(xi)) * hx), 1, solXfac, solYfac);
      }//end xi
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else {
    assert(false);
  }
}

void chkBoundaries(DM da, Vec vec) {
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

  const int solXfac = 1;
  const int solYfac = 1;

  if(dim == 1) {
    PetscScalar** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      assert( softEquals(arr[0][0], solution1D(0, solXfac)) );
    }
    if((xs + nx) == Nx) {
      assert( softEquals(arr[Nx - 1][0], solution1D(1, solXfac)) );
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        assert( softEquals(arr[yi][0][0] , solution2D(0, ((static_cast<long double>(yi)) * hy), solXfac, solYfac)) );
      }//end yi
    }
    if((xs + nx) == Nx) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        assert( softEquals(arr[yi][Nx - 1][0] , solution2D(1, ((static_cast<long double>(yi)) * hy), solXfac, solYfac)) );
      }//end yi
    }
    if(ys == 0) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        assert( softEquals(arr[0][xi][0] , solution2D(((static_cast<long double>(xi)) * hx), 0, solXfac, solYfac)) );
      }//end xi
    }
    if((ys + ny) == Ny) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        assert( softEquals(arr[Ny - 1][xi][0] , solution2D(((static_cast<long double>(xi)) * hx), 1, solXfac, solYfac)) );
      }//end xi
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else {
    assert(false);
  }

}



