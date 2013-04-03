
#include "gmg/include/mms.h"
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

  long double hy = 0;
  PetscInt nye = ny;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  }

  long double hz = 0;
  PetscInt nze = nz;
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  }

  PetscInt extraNumGpts = 0;
  //PetscOptionsGetInt(PETSC_NULL, "-extraGptsError", &extraNumGpts, PETSC_NULL);
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
                * shFnVals[nodeX][dofX][gX] );
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
    for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
      long double za = (static_cast<long double>(zi))*hz;
      for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
        long double ya = (static_cast<long double>(yi))*hy;
        for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
          long double xa = (static_cast<long double>(xi))*hx;
          for(int gZ = 0; gZ < numGaussPts; ++gZ) {
            long double zg = coordLocalToGlobal(gPt[gZ], za, hz);
            for(int gY = 0; gY < numGaussPts; ++gY) {
              long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
              for(int gX = 0; gX < numGaussPts; ++gX) {
                long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
                long double solVal = 0.0;
                for(int nodeZ = 0; nodeZ < 2; ++nodeZ) {
                  for(int nodeY = 0; nodeY < 2; ++nodeY) {
                    for(int nodeX = 0; nodeX < 2; ++nodeX) {
                      for(int dofZ = 0, d = 0; dofZ <= K; ++dofZ) {
                        for(int dofY = 0; dofY <= K; ++dofY) {
                          for(int dofX = 0; dofX <= K; ++dofX, ++d) {
                            solVal += ( (static_cast<long double>(arr3d[zi + nodeZ][yi + nodeY][xi + nodeX][d])) 
                                * shFnVals[nodeX][dofX][gX] * shFnVals[nodeY][dofY][gY] * shFnVals[nodeZ][dofZ][gZ] );
                          }//end dofX
                        }//end dofY
                      }//end dofZ
                    }//end nodeX
                  }//end nodeY
                }//end nodeZ
                long double err = solVal -  solution3D(xg, yg, zg);
                locErrSqr += ( gWt[gZ] * gWt[gY] * gWt[gX] * err * err );
              }//end gX
            }//end gY
          }//end gZ
        }//end xi
      }//end yi
    }//end zi
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
  if(dim > 2) {
    jac *= (hz * 0.5L);
  }

  long double result = sqrt(jac * globErrSqr);
  return result;
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

  long double hy = 0;
  PetscInt nye = ny;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  }

  long double hz = 0;
  PetscInt nze = nz;
  if(dim > 2) {
    hz = 1.0L/(static_cast<long double>(Nz - 1));
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
  }

  PetscInt extraNumGpts = 0;
  //PetscOptionsGetInt(PETSC_NULL, "-extraGptsRHS", &extraNumGpts, PETSC_NULL);
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
  PetscScalar**** arr3d = NULL;

  if(dim == 1) {
    DMDAVecGetArrayDOF(da, locRhs, &arr1d);
  } else if(dim == 2) {
    DMDAVecGetArrayDOF(da, locRhs, &arr2d);
  } else {
    DMDAVecGetArrayDOF(da, locRhs, &arr3d);
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
            arr1d[xi + node][dof] += ( gWt[g] * shFnVals[node][dof][g] * force1D(xg) );
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
                    arr2d[yi + nodeY][xi + nodeX][d] += ( gWt[gX] * gWt[gY] 
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
    for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
      long double za = (static_cast<long double>(zi))*hz;
      for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
        long double ya = (static_cast<long double>(yi))*hy;
        for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
          long double xa = (static_cast<long double>(xi))*hx;
          for(int nodeZ = 0; nodeZ < 2; ++nodeZ) {
            for(int nodeY = 0; nodeY < 2; ++nodeY) {
              for(int nodeX = 0; nodeX < 2; ++nodeX) {
                for(int dofZ = 0, d = 0; dofZ <= K; ++dofZ) {
                  for(int dofY = 0; dofY <= K; ++dofY) {
                    for(int dofX = 0; dofX <= K; ++dofX, ++d) {
                      for(int gZ = 0; gZ < numGaussPts; ++gZ) {
                        long double zg = coordLocalToGlobal(gPt[gZ], za, hz);
                        for(int gY = 0; gY < numGaussPts; ++gY) {
                          long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
                          for(int gX = 0; gX < numGaussPts; ++gX) {
                            long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
                            arr3d[zi + nodeZ][yi + nodeY][xi + nodeX][d] += ( gWt[gX] * gWt[gY] * gWt[gZ] 
                                * shFnVals[nodeX][dofX][gX] * shFnVals[nodeY][dofY][gY] * shFnVals[nodeZ][dofZ][gZ]
                                * force3D(xg, yg, zg) );
                          }//end gX
                        }//end gY
                      }//end gZ
                    }//end dofX
                  }//end dofY
                }//end dofZ
              }//end nodeX
            }//end nodeY
          }//end nodeZ
        }//end xi
      }//end yi
    }//end zi
  }

  if(dim == 1) {
    DMDAVecRestoreArrayDOF(da, locRhs, &arr1d);
  } else if(dim == 2) {
    DMDAVecRestoreArrayDOF(da, locRhs, &arr2d);
  } else {
    DMDAVecRestoreArrayDOF(da, locRhs, &arr3d);
  }

  VecZeroEntries(rhs);

  DMLocalToGlobalBegin(da, locRhs, ADD_VALUES, rhs);
  DMLocalToGlobalEnd(da, locRhs, ADD_VALUES, rhs);

  DMRestoreLocalVector(da, &locRhs);

  long double jac = hx * 0.5L;
  if(dim > 1) {
    jac *= (hy * 0.5L);
  }
  if(dim > 2) {
    jac *= (hz * 0.5L);
  }

  VecScale(rhs, jac);
}

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
  long double hy = 0;
  if(dim > 1) {
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz = 0;
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


