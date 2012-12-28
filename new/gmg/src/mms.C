
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#include <cassert>

long double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K) {
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

  PetscInt nxe = nx;
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }

  PetscInt extraNumGpts = 0;
  PetscOptionsGetInt(PETSC_NULL, "-extraGptsError", &extraNumGpts, PETSC_NULL);
  PetscInt numGaussPts = (2*K) + 3 + extraNumGpts;
  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  std::vector<std::vector<std::vector<long double> > > shFnVals(2);
  for(int node = 0; node < 2; ++node) {
    shFnVals[node].resize(K + 1);
    for(int dof = 0; dof <= K; ++dof) {
      (shFnVals[node][dof]).resize(numGaussPts);
      for(int g = 0; g < numGaussPts; ++g) {
        shFnVals[node][dof][g] = eval1DshFn(node, dof, K, coeffs, gPt[g]);
      }//end g
    }//end dof
  }//end node

  Vec locSol;
  DMGetLocalVector(da, &locSol);

  DMGlobalToLocalBegin(da, sol, INSERT_VALUES, locSol);
  DMGlobalToLocalEnd(da, sol, INSERT_VALUES, locSol);

  PetscScalar** arr1d = NULL;
  PetscScalar*** arr2d = NULL;
  PetscScalar**** arr3d = NULL;

  if(dim == 1) {
    DMDAVecGetArrayDOF(da, locSol, &arr1d);
  } else if(dim == 2) {
    DMDAVecGetArrayDOF(da, locSol, &arr2d);
  } else {
    DMDAVecGetArrayDOF(da, locSol, &arr3d);
  }

  const int solXfac = 1;

  long double locErrSqr = 0.0;
  if(dim == 1) {
    for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
      long double xa = (static_cast<long double>(xi))*hx;
      for(int gX = 0; gX < numGaussPts; ++gX) {
        long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
        long double solVal = 0.0;
        for(int nodeX = 0; nodeX < 2; ++nodeX) {
          for(int dofX = 0; dofX <= K; ++dofX) {
            solVal += ( (static_cast<long double>(arr1d[xi + nodeX][dofX])) 
                * myIntPow((0.5L * hx), dofX) * shFnVals[nodeX][dofX][gX] );
          }//end dofX
        }//end nodeX
        long double err = solVal -  solution1D(xg, solXfac);
        locErrSqr += ( gWt[gX] * err * err );
      }//end gX
    }//end xi
  } else if(dim == 2) {
    assert(false);
  } else {
    assert(false);
  }

  if(dim == 1) {
    DMDAVecRestoreArrayDOF(da, locSol, &arr1d);
  } else if(dim == 2) {
    DMDAVecRestoreArrayDOF(da, locSol, &arr2d);
  } else {
    DMDAVecRestoreArrayDOF(da, locSol, &arr3d);
  }

  DMRestoreLocalVector(da, &locSol);

  long double globErrSqr;
  MPI_Allreduce(&locErrSqr, &globErrSqr, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  long double scaling = hx*0.5L;

  long double result = sqrt(scaling*globErrSqr);

  return result;
}


