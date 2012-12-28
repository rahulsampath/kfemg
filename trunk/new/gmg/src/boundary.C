
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

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
  } else {
    assert(false);
  }

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
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        arr[yi][0][0] = 0.0;
      }//end yi
    }
    if((xs + nx) == Nx) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        arr[yi][Nx - 1][0] = 0.0;
      }//end yi
    }
    if(ys == 0) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        arr[0][xi][0] = 0.0;
      }//end xi
    }
    if((ys + ny) == Ny) {
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        arr[Ny - 1][xi][0] = 0.0;
      }//end xi
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  } else {
    PetscScalar**** arr; 
    DMDAVecGetArrayDOF(da, vec, &arr);
    if(xs == 0) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          arr[zi][yi][0][0] = 0.0;
        }//end yi
      }//end zi
    }
    if((xs + nx) == Nx) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
          arr[zi][yi][Nx - 1][0] = 0.0;
        }//end yi
      }//end zi
    }
    if(ys == 0) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          arr[zi][0][xi][0] = 0.0;
        }//end xi
      }//end zi
    }
    if((ys + ny) == Ny) {
      for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          arr[zi][Ny - 1][xi][0] = 0.0;
        }//end xi
      }//end zi
    }
    if(zs == 0) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          arr[0][yi][xi][0] = 0.0;
        }//end xi
      }//end yi
    }
    if((zs + nz) == Nz) {
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          arr[Nz - 1][yi][xi][0] = 0.0;
        }//end xi
      }//end yi
    }
    DMDAVecRestoreArrayDOF(da, vec, &arr);
  }
}



