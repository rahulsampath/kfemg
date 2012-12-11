
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

double computeError(DM da, Vec sol, std::vector<long long int>& coeffs, const int K) {
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

#ifdef DEBUG
  if(dim < 2) {
    assert(Ny == 1);
    assert(ys == 0);
    assert(ny == 1);
  }
  if(dim < 3) {
    assert(Nz == 1);
    assert(zs == 0);
    assert(nz == 1);
  }
#endif

  PetscInt nxe = nx;
  PetscInt nye = ny;
  PetscInt nze = nz;

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  long double hy = 0;
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  }
  long double hz = 0;
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  }

  //Is 2K + 3 sufficient? 2K + 2 is the minimum.
  int numGaussPts = (2*K) + 3;
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

  double locErrSqr = 0.0;
  if(dim == 1) {
    for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
      long double xa = (static_cast<long double>(xi))*hx;
      std::vector<PetscScalar> solVals(numGaussPts);
      for(int gX = 0; gX < numGaussPts; ++gX) {
        solVals[gX] = 0.0;
        for(int nodeX = 0; nodeX < 2; ++nodeX) {
          for(int dofX = 0; dofX <= K; ++dofX) {
            solVals[gX] += (arr1d[xi + nodeX][dofX] * shFnVals[nodeX][dofX][gX]);
          }//end dofX
        }//end nodeX
      }//end gX
    }//end xi
  } else if(dim == 2) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      long double ya = (static_cast<long double>(yi))*hy;
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        long double xa = (static_cast<long double>(xi))*hx;
      }//end xi
    }//end yi
  } else {
    for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
      long double za = (static_cast<long double>(zi))*hz;
      for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
        long double ya = (static_cast<long double>(yi))*hy;
        for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
          long double xa = (static_cast<long double>(xi))*hx;
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

  double globErrSqr;
  MPI_Allreduce(&locErrSqr, &globErrSqr, 1, MPI_DOUBLE,
      MPI_SUM, MPI_COMM_WORLD);

  long double scaling = hx*0.5L;
  if(dim > 1) {
    scaling *= (hy*0.5L);
  }
  if(dim > 2) {
    scaling *= (hz*0.5L);
  }

  double result = sqrt(scaling*globErrSqr);

  return result;
}

void computeRHS(DM da, std::vector<PetscInt>& lz, std::vector<PetscInt>& ly, std::vector<PetscInt>& lx,
    std::vector<PetscInt>& offsets, std::vector<long long int>& coeffs, const int K, Vec rhs) {
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

#ifdef DEBUG
  if(dim < 2) {
    assert(Ny == 1);
    assert(ys == 0);
    assert(ny == 1);
  }
  if(dim < 3) {
    assert(Nz == 1);
    assert(zs == 0);
    assert(nz == 1);
  }
#endif

  PetscInt nxe = nx;
  PetscInt nye = ny;
  PetscInt nze = nz;

  PetscInt numZnodes = 2;
  PetscInt numYnodes = 2;
  PetscInt numXnodes = 2;

  long double hx = 1.0L/(static_cast<long double>(Nx - 1));
  if((xs + nx) == Nx) {
    nxe = nx - 1;
  }
  long double hy = 0;
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
    hy = 1.0L/(static_cast<long double>(Ny - 1));
  } else {
    numYnodes = 1;
  }
  long double hz = 0;
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
    hz = 1.0L/(static_cast<long double>(Nz - 1));
  } else {
    numZnodes = 1;
  }

  unsigned int nodesPerElem = (1 << dim);

  std::vector<PetscInt> indices(nodesPerElem*dofsPerNode);
  std::vector<PetscScalar> vals(indices.size());

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int px = lx.size();
  int py = ly.size();

  int rk = rank/(px*py);
  int rj = (rank/px)%py;
  int ri = rank%px;

  int numGaussPts = (2*K) + 2;
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

  VecZeroEntries(rhs);

  //PERFORMANCE IMPROVEMENT: We could do a node-based assembly instead of the
  //following element-based assembly and avoid the communication.

  for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
    long double za = (static_cast<long double>(zi))*hz;
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      long double ya = (static_cast<long double>(yi))*hy;
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        long double xa = (static_cast<long double>(xi))*hx;
        for(PetscInt z = 0, i = 0; z < numZnodes; ++z) {
          PetscInt vk = (zi + z);
          int pk = rk;
          PetscInt vZs = zs;
          if(vk >= (zs + nz)) {
            ++pk;
            vZs += nz;
          }
          PetscInt zLoc = vk - vZs;
          for(PetscInt y = 0; y < numYnodes; ++y) {
            PetscInt vj = (yi + y);
            int pj = rj;
            PetscInt vYs = ys;
            if(vj >= (ys + ny)) {
              ++pj;
              vYs += ny;
            }
            PetscInt yLoc = vj - vYs;
            for(PetscInt x = 0; x < numXnodes; ++x) {
              PetscInt vi = (xi + x);
              int pi = ri;
              PetscInt vXs = xs;
              if(vi >= (xs + nx)) {
                ++pi;
                vXs += nx;
              }
              PetscInt xLoc = vi - vXs;
              int pid = (((pk*py) + pj)*px) + pi;
              PetscInt loc = (((zLoc*ly[pj]) + yLoc)*lx[pi]) + xLoc;
              PetscInt idBase = ((offsets[pid] + loc)*dofsPerNode);
              for(PetscInt d = 0; d < dofsPerNode; ++i, ++d) {
                indices[i] = idBase + d;
              }//end d
            }//end x
          }//end y
        }//end z
        if(dim == 1) {
          for(int node = 0, i = 0; node < 2; ++node) {
            for(int dof = 0; dof <= K; ++dof, ++i) {
              vals[i] = 0.0;
              for(int g = 0; g < numGaussPts; ++g) {
                long double xg = coordLocalToGlobal(gPt[g], xa, hx);
                vals[i] += ( gWt[g] * shFnVals[node][dof][g] * (__FORCE_1D__(xg)) );
              }//end g
            }//end dof
          }//end node
        } else if(dim == 2) {
          for(int nodeY = 0, i = 0; nodeY < 2; ++nodeY) {
            for(int nodeX = 0; nodeX < 2; ++nodeX) {
              for(int dofY = 0; dofY <= K; ++dofY) {
                for(int dofX = 0; dofX <= K; ++dofX, ++i) {
                  vals[i] = 0.0;
                  for(int gY = 0; gY < numGaussPts; ++gY) {
                    long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
                    for(int gX = 0; gX < numGaussPts; ++gX) {
                      long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
                      vals[i] += ( gWt[gY] * gWt[gX]  * shFnVals[nodeY][dofY][gY] *
                          shFnVals[nodeX][dofX][gX] * (__FORCE_2D__(xg, yg)) );
                    }//end gX
                  }//end gY
                }//end dofX
              }//end dofY
            }//end nodeX
          }//end nodeY
        } else {
          for(int nodeZ = 0, i = 0; nodeZ < 2; ++nodeZ) {
            for(int nodeY = 0; nodeY < 2; ++nodeY) {
              for(int nodeX = 0; nodeX < 2; ++nodeX) {
                for(int dofZ = 0; dofZ <= K; ++dofZ) {
                  for(int dofY = 0; dofY <= K; ++dofY) {
                    for(int dofX = 0; dofX <= K; ++dofX, ++i) {
                      vals[i] = 0.0;
                      for(int gZ = 0; gZ < numGaussPts; ++gZ) {
                        long double zg = coordLocalToGlobal(gPt[gZ], za, hz);
                        for(int gY = 0; gY < numGaussPts; ++gY) {
                          long double yg = coordLocalToGlobal(gPt[gY], ya, hy);
                          for(int gX = 0; gX < numGaussPts; ++gX) {
                            long double xg = coordLocalToGlobal(gPt[gX], xa, hx);
                            vals[i] += ( gWt[gZ] * gWt[gY] * gWt[gX] * shFnVals[nodeZ][dofZ][gZ] *
                                shFnVals[nodeY][dofY][gY] * shFnVals[nodeX][dofX][gX] * (__FORCE_3D__(xg, yg, zg)) );
                          }//end gX
                        }//end gY
                      }//end gZ
                    }//end dofX
                  }//end dofY
                }//end dofZ
              }//end nodeX
            }//end nodeY
          }//end nodeZ
        }
        VecSetValues(rhs, (indices.size()), &(indices[0]), &(vals[0]), ADD_VALUES);
      }//end xi
    }//end yi
  }//end zi

  VecAssemblyBegin(rhs);
  VecAssemblyEnd(rhs);

  long double scaling = hx*0.5L;
  if(dim > 1) {
    scaling *= (hy*0.5L);
  }
  if(dim > 2) {
    scaling *= (hz*0.5L);
  }

  VecScale(rhs, scaling);

  zeroBoundaries(da, rhs);
}


