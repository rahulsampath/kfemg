
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#include <cassert>

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

  PetscScalar** arr1d = NULL;
  PetscScalar*** arr2d = NULL;

  if(dim == 1) {
    DMDAVecGetArrayDOF(da, vec, &arr1d);
  } else if(dim == 2) {
    DMDAVecGetArrayDOF(da, vec, &arr2d);
  } else {
    assert(false);
  }

  if(dim == 1) {
    for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
      long double xa = (static_cast<long double>(xi))*hx;
      for(int d = 0; d <= K; ++d) {
        arr1d[xi][d] = solutionDerivative1D(xa, d);
      }//end dof
    }//end xi
  } else if(dim == 2) {
    for(PetscInt yi = ys; yi < (ys + ny); ++yi) {
      long double ya = (static_cast<long double>(yi))*hy;
      for(PetscInt xi = xs; xi < (xs + nx); ++xi) {
        long double xa = (static_cast<long double>(xi))*hx;
        for(int dofY = 0, d = 0; dofY <= K; ++dofY) {
          for(int dofX = 0; dofX <= K; ++dofX, ++d) {
            arr2d[yi][xi][d] = solutionDerivative2D(xa, ya, dofX, dofY);
          }//end dofX
        }//end dofY
      }//end xi
    }//end yi
  } else {
    assert(false);
  }

  if(dim == 1) {
    DMDAVecRestoreArrayDOF(da, vec, &arr1d);
  } else if(dim == 2) {
    DMDAVecRestoreArrayDOF(da, vec, &arr2d);
  } else {
    assert(false);
  }
}

void computeRHS(DM da, std::vector<long long int>& coeffs, const int K, Vec rhs) {
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

  long double hy;
  PetscInt nye = ny;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  }

  PetscInt extraNumGpts = 0;
  PetscOptionsGetInt(PETSC_NULL, "-extraGptsRHS", &extraNumGpts, PETSC_NULL);
  int numGaussPts = (2*K) + 2 + extraNumGpts;
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

  Vec locRhs;
  DMGetLocalVector(da, &locRhs);

  VecZeroEntries(locRhs);

  PetscScalar** arr1d = NULL;
  PetscScalar*** arr2d = NULL;

  if(dim == 1) {
    DMDAVecGetArrayDOF(da, locRhs, &arr1d);
  } else if(dim == 2) {
    DMDAVecGetArrayDOF(da, locRhs, &arr2d);
  } else {
    assert(false);
  }

  //PERFORMANCE IMPROVEMENT: We could do a node-based assembly instead of the
  //following element-based assembly and avoid the communication.

  if(dim == 1) {
    for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
      long double xa = (static_cast<long double>(xi))*hx;
      for(int node = 0; node < 2; ++node) {
        for(int dof = 0; dof <= K; ++dof) {
          for(int g = 0; g < numGaussPts; ++g) {
            long double xg = coordLocalToGlobal(gPt[g], xa, hx);
            arr1d[xi + node][dof] += ( gWt[g] * myIntPow((0.5L * hx), dof) 
                * shFnVals[node][dof][g] * force1D(xg) );
          }//end g
        }//end dof
      }//end node
    }//end xi
  } else if(dim == 2) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      long double ya = (static_cast<long double>(yi))*hy;
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        long double xa = (static_cast<long double>(xi))*hx;
        for(int nodeY = 0; nodeY < 2; ++nodeY) {
          for(int nodeX = 0; nodeX < 2; ++nodeX) {
            for(int dofY = 0, d = 0; dofY <= K; ++dofY) {
              for(int dofX = 0; dofX <= K; ++dofX, ++d) {
                for(int gY = 0; gY < numGaussPts; ++gY) {
                  long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
                  for(int gX = 0; gX < numGaussPts; ++gX) {
                    long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
                    arr2d[yi + nodeY][xi + nodeX][d] += ( gWt[gX] * gWt[gY] * myIntPow((0.5L * hx), dofX) * myIntPow((0.5L * hy), dofY)
                        * shFnVals[nodeX][dofX][gX] * shFnVals[nodeY][dofY][gY] * force2D(xg, yg) );
                  }//end gX
                }//end gY
              }//end dofX
            }//end dofY
          }//end nodeX
        }//end nodeY
      }//end xi
    }//end yi
  } else {
    assert(false);
  }

  if(dim == 1) {
    DMDAVecRestoreArrayDOF(da, locRhs, &arr1d);
  } else if(dim == 2) {
    DMDAVecRestoreArrayDOF(da, locRhs, &arr2d);
  } else {
    assert(false);
  }

  VecZeroEntries(rhs);

  DMLocalToGlobalBegin(da, locRhs, ADD_VALUES, rhs);
  DMLocalToGlobalEnd(da, locRhs, ADD_VALUES, rhs);

  DMRestoreLocalVector(da, &locRhs);

  long double jac = hx * 0.5L;
  if(dim > 1) {
    jac *= (hy * 0.5L);
  }

  VecScale(rhs, jac);
}

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

  long double hy;
  PetscInt nye = ny;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
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
        long double err = solVal -  solution1D(xg);
        locErrSqr += ( gWt[gX] * err * err );
      }//end gX
    }//end xi
  } else if(dim == 2) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      long double ya = (static_cast<long double>(yi))*hy;
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        long double xa = (static_cast<long double>(xi))*hx;
        for(int gY = 0; gY < numGaussPts; ++gY) {
          long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
          for(int gX = 0; gX < numGaussPts; ++gX) {
            long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
            long double solVal = 0.0;
            for(int nodeY = 0; nodeY < 2; ++nodeY) {
              for(int nodeX = 0; nodeX < 2; ++nodeX) {
                for(int dofY = 0, d = 0; dofY <= K; ++dofY) {
                  for(int dofX = 0; dofX <= K; ++dofX, ++d) {
                    solVal += ( (static_cast<long double>(arr2d[yi + nodeY][xi + nodeX][d])) 
                        * myIntPow((0.5L * hx), dofX) * myIntPow((0.5L * hy), dofY)
                        * shFnVals[nodeX][dofX][gX] * shFnVals[nodeY][dofY][gY] );
                  }//end dofX
                }//end dofY
              }//end nodeX
            }//end nodeY
            long double err = solVal -  solution2D(xg, yg);
            locErrSqr += ( gWt[gY] * gWt[gX] * err * err );
          }//end gX
        }//end gY
      }//end xi
    }//end yi
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

  long double jac = hx * 0.5L;
  if(dim > 1) {
    jac *= (hy * 0.5L);
  }

  long double result = sqrt(jac * globErrSqr);

  return result;
}


