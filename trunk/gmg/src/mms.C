
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void setSolution(DM da, Vec vec, const int K) {
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

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  long double hy;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz;
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  }

  PetscScalar** arr1d = NULL;
  PetscScalar*** arr2d = NULL;
  PetscScalar**** arr3d = NULL;

  if(dim == 1) {
    DMDAVecGetArrayDOF(da, vec, &arr1d);
  } else if(dim == 2) {
    DMDAVecGetArrayDOF(da, vec, &arr2d);
  } else {
    DMDAVecGetArrayDOF(da, vec, &arr3d);
  }

  if(dim == 1) {
    for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
      long double xa = (static_cast<long double>(xi))*hx;
      for(int d = 0; d <= K; ++d) {
        arr1d[xi][d] = myIntPow((0.5L * hx), d) * solutionDerivative1D(xa, d);
      }//end dof
    }//end xi
  } else if(dim == 2) {
    for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
      long double ya = (static_cast<long double>(yi))*hy;
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        long double xa = (static_cast<long double>(xi))*hx;
        for(int dofY = 0, d = 0; dofY <= K; ++dofY) {
          for(int dofX = 0; dofX <= K; ++dofX, ++d) {
            arr2d[yi][xi][d] = myIntPow((0.5L * hx), dofX) * myIntPow((0.5L * hy), dofY)
              * solutionDerivative2D(xa, ya, dofX, dofY);
          }//end dofX
        }//end dofY
      }//end xi
    }//end yi
  } else {
    for(PetscInt zi = zs; zi < (zs + nz); ++zi) {
      long double za = (static_cast<long double>(zi))*hz;
      for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
        long double ya = (static_cast<long double>(yi))*hy;
        for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
          long double xa = (static_cast<long double>(xi))*hx;
          for(int dofZ = 0, d = 0; dofZ <= K; ++dofZ) {
            for(int dofY = 0; dofY <= K; ++dofY) {
              for(int dofX = 0; dofX <= K; ++dofX, ++d) {
                arr3d[zi][yi][xi][d] = myIntPow((0.5L * hx), dofX) * myIntPow((0.5L * hy), dofY) 
                  * myIntPow((0.5L * hz), dofZ) * solutionDerivative3D(xa, ya, za, dofX, dofY, dofZ);
              }//end dofX
            }//end dofY
          }//end dofZ
        }//end xi
      }//end yi
    }//end zi
  }

  if(dim == 1) {
    DMDAVecRestoreArrayDOF(da, vec, &arr1d);
  } else if(dim == 2) {
    DMDAVecRestoreArrayDOF(da, vec, &arr2d);
  } else {
    DMDAVecRestoreArrayDOF(da, vec, &arr3d);
  }
}


