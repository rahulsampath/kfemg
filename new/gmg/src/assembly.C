
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void computeKmat(Mat Kmat, DM da, std::vector<std::vector<long double> >& elemMat) {
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

  PetscInt nxe = nx;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }

  MatZeroEntries(Kmat);

  if(dim == 1) {
    for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
      for(int rNode = 0, r = 0; rNode < 2; ++rNode) {
        for(int rDof = 0; rDof < dofsPerNode; ++rDof, ++r) {
          MatStencil row;
          row.i = xi + rNode;
          row.c = rDof;
          for(int cNode = 0, c = 0; cNode < 2; ++cNode) {
            for(int cDof = 0; cDof < dofsPerNode; ++cDof, ++c) {
              MatStencil col;
              col.i = xi + cNode;
              col.c = cDof;
              PetscScalar val = elemMat[r][c];
              MatSetValuesStencil(Kmat, 1, &row, 1, &col, &val, ADD_VALUES);
            }//end cDof
          }//end cNode
        }//end rDof
      }//end rNode
    }//end xi
  } else {
    assert(false);
  }

  MatAssemblyBegin(Kmat, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FLUSH_ASSEMBLY);
}


