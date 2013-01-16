
#include <iostream>
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void assembleKmat(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<Mat>& Kmat, std::vector<DM>& da, int K, std::vector<long long int>& coeffs, 
    std::vector<unsigned long long int>& factorialsList, bool print) {
  for(int lev = 0; lev < (Kmat.size()); ++lev, print = false) {
    if(Kmat[lev] != NULL) {
      std::vector<std::vector<long double> > elemMat;
      if(dim == 1) {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        createPoisson1DelementMatrix(factorialsList, K, coeffs, hx, elemMat, print);
      } else if(dim == 2) {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        long double hy = 1.0L/(static_cast<long double>(Ny[lev] - 1));
        createPoisson2DelementMatrix(factorialsList, K, coeffs, hy, hx, elemMat, print);
      } else {
        long double hx = 1.0L/(static_cast<long double>(Nx[lev] - 1));
        long double hy = 1.0L/(static_cast<long double>(Ny[lev] - 1));
        long double hz = 1.0L/(static_cast<long double>(Nz[lev] - 1));
        createPoisson3DelementMatrix(factorialsList, K, coeffs, hz, hy, hx, elemMat, print);
      }
      computeKmat(Kmat[lev], da[lev], elemMat);
    }
  }//end lev
}

void buildKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, bool print) {
  Kmat.clear();
  Kmat.resize(da.size(), NULL);
  for(int lev = 0; lev < (da.size()); ++lev) {
    if(da[lev] != NULL) {
      DMCreateMatrix(da[lev], MATAIJ, &(Kmat[lev]));
      PetscInt sz;
      MatGetSize(Kmat[lev], &sz, PETSC_NULL);
      if(print) {
        std::cout<<"Lev = "<<lev<<", Kmat size = "<<sz<<std::endl;
      }
    }
  }//end lev
}

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
  PetscInt nye = ny;
  if(dim > 1) {
    if((ys + ny) == Ny) {
      nye = ny - 1;
    }
  }
  PetscInt nze = nz;
  if(dim > 2) {
    if((zs + nz) == Nz) {
      nze = nz - 1;
    }
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
  } else if(dim == 2) {
    for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
      for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
        for(int rNodeY = 0, r = 0; rNodeY < 2; ++rNodeY) {
          for(int rNodeX = 0; rNodeX < 2; ++rNodeX) {
            for(int rDof = 0; rDof < dofsPerNode; ++rDof, ++r) {
              MatStencil row;
              row.j = yi + rNodeY;
              row.i = xi + rNodeX;
              row.c = rDof;
              for(int cNodeY = 0, c = 0; cNodeY < 2; ++cNodeY) {
                for(int cNodeX = 0; cNodeX < 2; ++cNodeX) {
                  for(int cDof = 0; cDof < dofsPerNode; ++cDof, ++c) {
                    MatStencil col;
                    col.j = yi + cNodeY;
                    col.i = xi + cNodeX;
                    col.c = cDof;
                    PetscScalar val = elemMat[r][c];
                    MatSetValuesStencil(Kmat, 1, &row, 1, &col, &val, ADD_VALUES);
                  }//end cDof
                }//end cNodeX
              }//end cNodeY
            }//end rDof
          }//end rNodeX
        }//end rNodeY
      }//end xi
    }//end yi
  } else {
    for(PetscInt zi = zs; zi < (zs + nze); ++zi) {
      for(PetscInt yi = ys; yi < (ys + nye); ++yi) {
        for(PetscInt xi = xs; xi < (xs + nxe); ++xi) {
          for(int rNodeZ = 0, r = 0; rNodeZ < 2; ++rNodeZ) {
            for(int rNodeY = 0; rNodeY < 2; ++rNodeY) {
              for(int rNodeX = 0; rNodeX < 2; ++rNodeX) {
                for(int rDof = 0; rDof < dofsPerNode; ++rDof, ++r) {
                  MatStencil row;
                  row.k = zi + rNodeZ;
                  row.j = yi + rNodeY;
                  row.i = xi + rNodeX;
                  row.c = rDof;
                  for(int cNodeZ = 0, c = 0; cNodeZ < 2; ++cNodeZ) {
                    for(int cNodeY = 0; cNodeY < 2; ++cNodeY) {
                      for(int cNodeX = 0; cNodeX < 2; ++cNodeX) {
                        for(int cDof = 0; cDof < dofsPerNode; ++cDof, ++c) {
                          MatStencil col;
                          col.k = zi + cNodeZ;
                          col.j = yi + cNodeY;
                          col.i = xi + cNodeX;
                          col.c = cDof;
                          PetscScalar val = elemMat[r][c];
                          MatSetValuesStencil(Kmat, 1, &row, 1, &col, &val, ADD_VALUES);
                        }//end cDof
                      }//end cNodeX
                    }//end cNodeY
                  }//end cNodeZ
                }//end rDof
              }//end rNodeX
            }//end rNodeY
          }//end rNodeZ
        }//end xi
      }//end yi
    }//end zi
  }

  MatAssemblyBegin(Kmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Kmat, MAT_FINAL_ASSEMBLY);
}


