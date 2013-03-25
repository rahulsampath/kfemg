
#include "gmg/include/fd.h"
#include <vector>

#ifdef DEBUG
#include <cassert>
#endif

void applyFD(DM da, int K, Vec in, Vec out) {
  PetscInt dim;
  DMDAGetInfo(da, &dim, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

#ifdef DEBUG
  assert(nx >= 5);
  if(dim > 1) {
    assert(ny >= 5);
  }
  if(dim > 2) {
    assert(nz >= 5);
  }
#endif

  if(dim == 1) {
    PetscScalar** inArr;
    PetscScalar** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);
    outArr[xs][K] = -((25.0*inArr[xs][K - 1]) - (48.0*inArr[xs + 1][K - 1]) + 
        (36.0*inArr[xs + 2][K - 1]) - (16.0*inArr[xs + 3][K - 1])
        + (3.0*inArr[xs + 4][K - 1]))/24.0;
    outArr[xs + 1][K] = ((-3.0*inArr[xs][K - 1]) - (10.0*inArr[xs + 1][K - 1]) +
        (18.0*inArr[xs + 2][K - 1]) - (6.0*inArr[xs + 3][K - 1])
        + inArr[xs + 4][K - 1])/24.0;
    for(int xi = xs + 2; xi < (xs + nx - 2); ++xi) {
      outArr[xi][K] = (-inArr[xi + 2][K - 1] + (8.0*inArr[xi + 1][K - 1]) - 
          (8.0*inArr[xi - 1][K - 1]) + inArr[xi - 2][K - 1])/24.0;
    }//end xi
    outArr[xs + nx - 2][K] = -((-3.0*inArr[xs + nx - 1][K - 1]) - (10.0*inArr[xs + nx - 2][K - 1]) +
        (18.0*inArr[xs + nx - 3][K - 1]) - (6.0*inArr[xs + nx - 4][K - 1])
        + inArr[xs + nx - 5][K - 1])/24.0;
    outArr[xs + nx - 1][K] = ((25.0*inArr[xs + nx - 1][K - 1]) - (48.0*inArr[xs + nx - 2][K - 1]) + 
        (36.0*inArr[xs + nx - 3][K - 1]) - (16.0*inArr[xs + nx - 4][K - 1])
        + (3.0*inArr[xs + nx - 5][K - 1]))/24.0;
    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  } else if(dim == 2) {
    PetscScalar*** inArr;
    PetscScalar*** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);
    //dx
    for(int yi = ys; yi < (ys + ny); ++yi) {
      for(int d = 0; d < K; ++d) {
        int outDof = (d*(K + 1)) + K;
        int inDof = (d*(K + 1)) + K - 1;
        outArr[yi][xs][outDof] = -((25.0*inArr[yi][xs][inDof]) - (48.0*inArr[yi][xs + 1][inDof]) + 
            (36.0*inArr[yi][xs + 2][inDof]) - (16.0*inArr[yi][xs + 3][inDof])
            + (3.0*inArr[yi][xs + 4][inDof]))/24.0;
        outArr[yi][xs + 1][outDof] = ((-3.0*inArr[yi][xs][inDof]) - (10.0*inArr[yi][xs + 1][inDof]) +
            (18.0*inArr[yi][xs + 2][inDof]) - (6.0*inArr[yi][xs + 3][inDof])
            + inArr[yi][xs + 4][inDof])/24.0;
        for(int xi = xs + 2; xi < (xs + nx - 2); ++xi) {
          outArr[yi][xi][outDof] = (-inArr[yi][xi + 2][inDof] + (8.0*inArr[yi][xi + 1][inDof]) - 
              (8.0*inArr[yi][xi - 1][inDof]) + inArr[yi][xi - 2][inDof])/24.0;
        }//end xi
        outArr[yi][xs + nx - 2][outDof] = -((-3.0*inArr[yi][xs + nx - 1][inDof]) - (10.0*inArr[yi][xs + nx - 2][inDof]) +
            (18.0*inArr[yi][xs + nx - 3][inDof]) - (6.0*inArr[yi][xs + nx - 4][inDof])
            + inArr[yi][xs + nx - 5][inDof])/24.0;
        outArr[yi][xs + nx - 1][outDof] = ((25.0*inArr[yi][xs + nx - 1][inDof]) - (48.0*inArr[yi][xs + nx - 2][inDof]) + 
            (36.0*inArr[yi][xs + nx - 3][inDof]) - (16.0*inArr[yi][xs + nx - 4][inDof])
            + (3.0*inArr[yi][xs + nx - 5][inDof]))/24.0;
      }//end d
    }//end yi
    //dy
    for(int xi = xs; xi < (xs + nx); ++xi) {
      for(int d = 0; d < K; ++d) {
        int outDof = (K*(K + 1)) + d;
        int inDof = ((K - 1)*(K + 1)) + d;
        outArr[ys][xi][outDof] = -((25.0*inArr[ys][xi][inDof]) - (48.0*inArr[ys + 1][xi][inDof]) + 
            (36.0*inArr[ys + 2][xi][inDof]) - (16.0*inArr[ys + 3][xi][inDof])
            + (3.0*inArr[ys + 4][xi][inDof]))/24.0;
        outArr[ys + 1][xi][outDof] = ((-3.0*inArr[ys][xi][inDof]) - (10.0*inArr[ys + 1][xi][inDof]) +
            (18.0*inArr[ys + 2][xi][inDof]) - (6.0*inArr[ys + 3][xi][inDof])
            + inArr[ys + 4][xi][inDof])/24.0;
        for(int yi = ys + 2; yi < (ys + ny - 2); ++yi) {
          outArr[yi][xi][outDof] = (-inArr[yi + 2][xi][inDof] + (8.0*inArr[yi + 1][xi][inDof]) - 
              (8.0*inArr[yi - 1][xi][inDof]) + inArr[yi - 2][xi][inDof])/24.0;
        }//end yi
        outArr[ys + ny - 2][xi][outDof] = -((-3.0*inArr[ys + ny - 1][xi][inDof]) - (10.0*inArr[ys + ny - 2][xi][inDof]) +
            (18.0*inArr[ys + ny - 3][xi][inDof]) - (6.0*inArr[ys + ny - 4][xi][inDof])
            + inArr[ys + ny - 5][xi][inDof])/24.0;
        outArr[ys + ny - 1][xi][outDof] = ((25.0*inArr[ys + ny - 1][xi][inDof]) - (48.0*inArr[ys + ny - 2][xi][inDof]) + 
            (36.0*inArr[ys + ny - 3][xi][inDof]) - (16.0*inArr[ys + ny - 4][xi][inDof])
            + (3.0*inArr[ys + ny - 5][xi][inDof]))/24.0;
      }//end d
    }//end xi
    //dxdy
    for(int yi = ys; yi < (ys + ny); ++yi) {
      int outDof = (K*(K + 1)) + K;
      int inDof = (K*(K + 1)) + K - 1;
      outArr[yi][xs][outDof] = -((25.0*outArr[yi][xs][inDof]) - (48.0*outArr[yi][xs + 1][inDof]) + 
          (36.0*outArr[yi][xs + 2][inDof]) - (16.0*outArr[yi][xs + 3][inDof])
          + (3.0*outArr[yi][xs + 4][inDof]))/24.0;
      outArr[yi][xs + 1][outDof] = ((-3.0*outArr[yi][xs][inDof]) - (10.0*outArr[yi][xs + 1][inDof]) +
          (18.0*outArr[yi][xs + 2][inDof]) - (6.0*outArr[yi][xs + 3][inDof])
          + outArr[yi][xs + 4][inDof])/24.0;
      for(int xi = xs + 2; xi < (xs + nx - 2); ++xi) {
        outArr[yi][xi][outDof] = (-outArr[yi][xi + 2][inDof] + (8.0*outArr[yi][xi + 1][inDof]) - 
            (8.0*outArr[yi][xi - 1][inDof]) + outArr[yi][xi - 2][inDof])/24.0;
      }//end xi
      outArr[yi][xs + nx - 2][outDof] = -((-3.0*outArr[yi][xs + nx - 1][inDof]) - (10.0*outArr[yi][xs + nx - 2][inDof]) +
          (18.0*outArr[yi][xs + nx - 3][inDof]) - (6.0*outArr[yi][xs + nx - 4][inDof])
          + outArr[yi][xs + nx - 5][inDof])/24.0;
      outArr[yi][xs + nx - 1][outDof] = ((25.0*outArr[yi][xs + nx - 1][inDof]) - (48.0*outArr[yi][xs + nx - 2][inDof]) + 
          (36.0*outArr[yi][xs + nx - 3][inDof]) - (16.0*outArr[yi][xs + nx - 4][inDof])
          + (3.0*outArr[yi][xs + nx - 5][inDof]))/24.0;
    }//end yi
    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  } else {
    PetscScalar**** inArr;
    PetscScalar**** outArr;
    DMDAVecGetArrayDOF(da, in, &inArr);
    DMDAVecGetArrayDOF(da, out, &outArr);
    //dx
    for(int zi = zs; zi < (zs + nz); ++zi) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int dz = 0; dz < K; ++dz) {
          for(int dy = 0; dy < K; ++dy) {
            int outDof = (((dz*(K + 1)) + dy)*(K + 1)) + K;
            int inDof = (((dz*(K + 1)) + dy)*(K + 1)) + K - 1;
            outArr[zi][yi][xs][outDof] = -((25.0*inArr[zi][yi][xs][inDof]) - (48.0*inArr[zi][yi][xs + 1][inDof]) + 
                (36.0*inArr[zi][yi][xs + 2][inDof]) - (16.0*inArr[zi][yi][xs + 3][inDof])
                + (3.0*inArr[zi][yi][xs + 4][inDof]))/24.0;
            outArr[zi][yi][xs + 1][outDof] = ((-3.0*inArr[zi][yi][xs][inDof]) - (10.0*inArr[zi][yi][xs + 1][inDof]) +
                (18.0*inArr[zi][yi][xs + 2][inDof]) - (6.0*inArr[zi][yi][xs + 3][inDof])
                + inArr[zi][yi][xs + 4][inDof])/24.0;
            for(int xi = xs + 2; xi < (xs + nx - 2); ++xi) {
              outArr[zi][yi][xi][outDof] = (-inArr[zi][yi][xi + 2][inDof] + (8.0*inArr[zi][yi][xi + 1][inDof]) - 
                  (8.0*inArr[zi][yi][xi - 1][inDof]) + inArr[zi][yi][xi - 2][inDof])/24.0;
            }//end xi
            outArr[zi][yi][xs + nx - 2][outDof] = -((-3.0*inArr[zi][yi][xs + nx - 1][inDof]) - (10.0*inArr[zi][yi][xs + nx - 2][inDof]) +
                (18.0*inArr[zi][yi][xs + nx - 3][inDof]) - (6.0*inArr[zi][yi][xs + nx - 4][inDof])
                + inArr[zi][yi][xs + nx - 5][inDof])/24.0;
            outArr[zi][yi][xs + nx - 1][outDof] = ((25.0*inArr[zi][yi][xs + nx - 1][inDof]) - (48.0*inArr[zi][yi][xs + nx - 2][inDof]) + 
                (36.0*inArr[zi][yi][xs + nx - 3][inDof]) - (16.0*inArr[zi][yi][xs + nx - 4][inDof])
                + (3.0*inArr[zi][yi][xs + nx - 5][inDof]))/24.0;
          }//end dy
        }//end dz
      }//end yi
    }//end zi
    //dy
    for(int zi = zs; zi < (zs + nz); ++zi) {
      for(int xi = xs; xi < (xs + nx); ++xi) {
        for(int dz = 0; dz < K; ++dz) {
          for(int dx = 0; dx < K; ++dx) {
            int outDof = (((dz*(K + 1)) + K)*(K + 1)) + dx;
            int inDof = (((dz*(K + 1)) + (K - 1))*(K + 1)) + dx;
            outArr[zi][ys][xi][outDof] = -((25.0*inArr[zi][ys][xi][inDof]) - (48.0*inArr[zi][ys + 1][xi][inDof]) + 
                (36.0*inArr[zi][ys + 2][xi][inDof]) - (16.0*inArr[zi][ys + 3][xi][inDof])
                + (3.0*inArr[zi][ys + 4][xi][inDof]))/24.0;
            outArr[zi][ys + 1][xi][outDof] = ((-3.0*inArr[zi][ys][xi][inDof]) - (10.0*inArr[zi][ys + 1][xi][inDof]) +
                (18.0*inArr[zi][ys + 2][xi][inDof]) - (6.0*inArr[zi][ys + 3][xi][inDof])
                + inArr[zi][ys + 4][xi][inDof])/24.0;
            for(int yi = ys + 2; yi < (ys + ny - 2); ++yi) {
              outArr[zi][yi][xi][outDof] = (-inArr[zi][yi + 2][xi][inDof] + (8.0*inArr[zi][yi + 1][xi][inDof]) - 
                  (8.0*inArr[zi][yi - 1][xi][inDof]) + inArr[zi][yi - 2][xi][inDof])/24.0;
            }//end yi
            outArr[zi][ys + ny - 2][xi][outDof] = -((-3.0*inArr[zi][ys + ny - 1][xi][inDof]) - (10.0*inArr[zi][ys + ny - 2][xi][inDof]) +
                (18.0*inArr[zi][ys + ny - 3][xi][inDof]) - (6.0*inArr[zi][ys + ny - 4][xi][inDof])
                + inArr[zi][ys + ny - 5][xi][inDof])/24.0;
            outArr[zi][ys + ny - 1][xi][outDof] = ((25.0*inArr[zi][ys + ny - 1][xi][inDof]) - (48.0*inArr[zi][ys + ny - 2][xi][inDof]) + 
                (36.0*inArr[zi][ys + ny - 3][xi][inDof]) - (16.0*inArr[zi][ys + ny - 4][xi][inDof])
                + (3.0*inArr[zi][ys + ny - 5][xi][inDof]))/24.0;
          }//end dx
        }//end dz
      }//end xi
    }//end zi
    //dz
    for(int yi = ys; yi < (ys + ny); ++yi) {
      for(int xi = xs; xi < (xs + nx); ++xi) {
        for(int dy = 0; dy < K; ++dy) {
          for(int dx = 0; dx < K; ++dx) {
            int outDof = (((K*(K + 1)) + dy)*(K + 1)) + dx;
            int inDof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + dx;
            outArr[zs][yi][xi][outDof] = -((25.0*inArr[zs][yi][xi][inDof]) - (48.0*inArr[zs + 1][yi][xi][inDof]) + 
                (36.0*inArr[zs + 2][yi][xi][inDof]) - (16.0*inArr[zs + 3][yi][xi][inDof])
                + (3.0*inArr[zs + 4][yi][xi][inDof]))/24.0;
            outArr[zs + 1][yi][xi][outDof] = ((-3.0*inArr[zs][yi][xi][inDof]) - (10.0*inArr[zs + 1][yi][xi][inDof]) +
                (18.0*inArr[zs + 2][yi][xi][inDof]) - (6.0*inArr[zs + 3][yi][xi][inDof])
                + inArr[zs + 4][yi][xi][inDof])/24.0;
            for(int zi = zs + 2; zi < (zs + nz - 2); ++zi) {
              outArr[zi][yi][xi][outDof] = (-inArr[zi + 2][yi][xi][inDof] + (8.0*inArr[zi + 1][yi][xi][inDof]) - 
                  (8.0*inArr[zi - 1][yi][xi][inDof]) + inArr[zi - 2][yi][xi][inDof])/24.0;
            }//end zi
            outArr[zs + nz - 2][yi][xi][outDof] = -((-3.0*inArr[zs + nz - 1][yi][xi][inDof]) - (10.0*inArr[zs + nz - 2][yi][xi][inDof]) +
                (18.0*inArr[zs + nz - 3][yi][xi][inDof]) - (6.0*inArr[zs + nz - 4][yi][xi][inDof])
                + inArr[zs + nz - 5][yi][xi][inDof])/24.0;
            outArr[zs + nz - 1][yi][xi][outDof] = ((25.0*inArr[zs + nz - 1][yi][xi][inDof]) - (48.0*inArr[zs + nz - 2][yi][xi][inDof]) + 
                (36.0*inArr[zs + nz - 3][yi][xi][inDof]) - (16.0*inArr[zs + nz - 4][yi][xi][inDof])
                + (3.0*inArr[zs + nz - 5][yi][xi][inDof]))/24.0;
          }//end dx
        }//end dy
      }//end xi
    }//end yi
    //dxdy
    for(int zi = zs; zi < (zs + nz); ++zi) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        for(int dz = 0; dz < K; ++dz) {
          int outDof = (((dz*(K + 1)) + K)*(K + 1)) + K;
          int inDof = (((dz*(K + 1)) + K)*(K + 1)) + K - 1;
          outArr[zi][yi][xs][outDof] = -((25.0*outArr[zi][yi][xs][inDof]) - (48.0*outArr[zi][yi][xs + 1][inDof]) + 
              (36.0*outArr[zi][yi][xs + 2][inDof]) - (16.0*outArr[zi][yi][xs + 3][inDof])
              + (3.0*outArr[zi][yi][xs + 4][inDof]))/24.0;
          outArr[zi][yi][xs + 1][outDof] = ((-3.0*outArr[zi][yi][xs][inDof]) - (10.0*outArr[zi][yi][xs + 1][inDof]) +
              (18.0*outArr[zi][yi][xs + 2][inDof]) - (6.0*outArr[zi][yi][xs + 3][inDof])
              + outArr[zi][yi][xs + 4][inDof])/24.0;
          for(int xi = xs + 2; xi < (xs + nx - 2); ++xi) {
            outArr[zi][yi][xi][outDof] = (-outArr[zi][yi][xi + 2][inDof] + (8.0*outArr[zi][yi][xi + 1][inDof]) - 
                (8.0*outArr[zi][yi][xi - 1][inDof]) + outArr[zi][yi][xi - 2][inDof])/24.0;
          }//end xi
          outArr[zi][yi][xs + nx - 2][outDof] = -((-3.0*outArr[zi][yi][xs + nx - 1][inDof]) - (10.0*outArr[zi][yi][xs + nx - 2][inDof]) +
              (18.0*outArr[zi][yi][xs + nx - 3][inDof]) - (6.0*outArr[zi][yi][xs + nx - 4][inDof])
              + outArr[zi][yi][xs + nx - 5][inDof])/24.0;
          outArr[zi][yi][xs + nx - 1][outDof] = ((25.0*outArr[zi][yi][xs + nx - 1][inDof]) - (48.0*outArr[zi][yi][xs + nx - 2][inDof]) + 
              (36.0*outArr[zi][yi][xs + nx - 3][inDof]) - (16.0*outArr[zi][yi][xs + nx - 4][inDof])
              + (3.0*outArr[zi][yi][xs + nx - 5][inDof]))/24.0;
        }//end dz
      }//end yi
    }//end zi
    //dydz
    for(int zi = zs; zi < (zs + nz); ++zi) {
      for(int xi = xs; xi < (xs + nx); ++xi) {
        for(int dx = 0; dx < K; ++dx) {
          int outDof = (((K*(K + 1)) + K)*(K + 1)) + dx;
          int inDof = (((K*(K + 1)) + (K - 1))*(K + 1)) + dx;
          outArr[zi][ys][xi][outDof] = -((25.0*outArr[zi][ys][xi][inDof]) - (48.0*outArr[zi][ys + 1][xi][inDof]) + 
              (36.0*outArr[zi][ys + 2][xi][inDof]) - (16.0*outArr[zi][ys + 3][xi][inDof])
              + (3.0*outArr[zi][ys + 4][xi][inDof]))/24.0;
          outArr[zi][ys + 1][xi][outDof] = ((-3.0*outArr[zi][ys][xi][inDof]) - (10.0*outArr[zi][ys + 1][xi][inDof]) +
              (18.0*outArr[zi][ys + 2][xi][inDof]) - (6.0*outArr[zi][ys + 3][xi][inDof])
              + outArr[zi][ys + 4][xi][inDof])/24.0;
          for(int yi = ys + 2; yi < (ys + ny - 2); ++yi) {
            outArr[zi][yi][xi][outDof] = (-outArr[zi][yi + 2][xi][inDof] + (8.0*outArr[zi][yi + 1][xi][inDof]) - 
                (8.0*outArr[zi][yi - 1][xi][inDof]) + outArr[zi][yi - 2][xi][inDof])/24.0;
          }//end yi
          outArr[zi][ys + ny - 2][xi][outDof] = -((-3.0*outArr[zi][ys + ny - 1][xi][inDof]) - (10.0*outArr[zi][ys + ny - 2][xi][inDof]) +
              (18.0*outArr[zi][ys + ny - 3][xi][inDof]) - (6.0*outArr[zi][ys + ny - 4][xi][inDof])
              + outArr[zi][ys + ny - 5][xi][inDof])/24.0;
          outArr[zi][ys + ny - 1][xi][outDof] = ((25.0*outArr[zi][ys + ny - 1][xi][inDof]) - (48.0*outArr[zi][ys + ny - 2][xi][inDof]) + 
              (36.0*outArr[zi][ys + ny - 3][xi][inDof]) - (16.0*outArr[zi][ys + ny - 4][xi][inDof])
              + (3.0*outArr[zi][ys + ny - 5][xi][inDof]))/24.0;
        }//end dx
      }//end xi
    }//end zi
    //dzdx
    for(int yi = ys; yi < (ys + ny); ++yi) {
      for(int xi = xs; xi < (xs + nx); ++xi) {
        for(int dy = 0; dy < K; ++dy) {
          int outDof = (((K*(K + 1)) + dy)*(K + 1)) + K;
          int inDof = ((((K - 1)*(K + 1)) + dy)*(K + 1)) + K;
          outArr[zs][yi][xi][outDof] = -((25.0*outArr[zs][yi][xi][inDof]) - (48.0*outArr[zs + 1][yi][xi][inDof]) + 
              (36.0*outArr[zs + 2][yi][xi][inDof]) - (16.0*outArr[zs + 3][yi][xi][inDof])
              + (3.0*outArr[zs + 4][yi][xi][inDof]))/24.0;
          outArr[zs + 1][yi][xi][outDof] = ((-3.0*outArr[zs][yi][xi][inDof]) - (10.0*outArr[zs + 1][yi][xi][inDof]) +
              (18.0*outArr[zs + 2][yi][xi][inDof]) - (6.0*outArr[zs + 3][yi][xi][inDof])
              + outArr[zs + 4][yi][xi][inDof])/24.0;
          for(int zi = zs + 2; zi < (zs + nz - 2); ++zi) {
            outArr[zi][yi][xi][outDof] = (-outArr[zi + 2][yi][xi][inDof] + (8.0*outArr[zi + 1][yi][xi][inDof]) - 
                (8.0*outArr[zi - 1][yi][xi][inDof]) + outArr[zi - 2][yi][xi][inDof])/24.0;
          }//end zi
          outArr[zs + nz - 2][yi][xi][outDof] = -((-3.0*outArr[zs + nz - 1][yi][xi][inDof]) - (10.0*outArr[zs + nz - 2][yi][xi][inDof]) +
              (18.0*outArr[zs + nz - 3][yi][xi][inDof]) - (6.0*outArr[zs + nz - 4][yi][xi][inDof])
              + outArr[zs + nz - 5][yi][xi][inDof])/24.0;
          outArr[zs + nz - 1][yi][xi][outDof] = ((25.0*outArr[zs + nz - 1][yi][xi][inDof]) - (48.0*outArr[zs + nz - 2][yi][xi][inDof]) + 
              (36.0*outArr[zs + nz - 3][yi][xi][inDof]) - (16.0*outArr[zs + nz - 4][yi][xi][inDof])
              + (3.0*outArr[zs + nz - 5][yi][xi][inDof]))/24.0;
        }//end dy
      }//end xi
    }//end yi
    //dxdydz
    for(int zi = zs; zi < (zs + nz); ++zi) {
      for(int yi = ys; yi < (ys + ny); ++yi) {
        int outDof = (((K*(K + 1)) + K)*(K + 1)) + K;
        int inDof = (((K*(K + 1)) + K)*(K + 1)) + K - 1;
        outArr[zi][yi][xs][outDof] = -((25.0*outArr[zi][yi][xs][inDof]) - (48.0*outArr[zi][yi][xs + 1][inDof]) + 
            (36.0*outArr[zi][yi][xs + 2][inDof]) - (16.0*outArr[zi][yi][xs + 3][inDof])
            + (3.0*outArr[zi][yi][xs + 4][inDof]))/24.0;
        outArr[zi][yi][xs + 1][outDof] = ((-3.0*outArr[zi][yi][xs][inDof]) - (10.0*outArr[zi][yi][xs + 1][inDof]) +
            (18.0*outArr[zi][yi][xs + 2][inDof]) - (6.0*outArr[zi][yi][xs + 3][inDof])
            + outArr[zi][yi][xs + 4][inDof])/24.0;
        for(int xi = xs + 2; xi < (xs + nx - 2); ++xi) {
          outArr[zi][yi][xi][outDof] = (-outArr[zi][yi][xi + 2][inDof] + (8.0*outArr[zi][yi][xi + 1][inDof]) - 
              (8.0*outArr[zi][yi][xi - 1][inDof]) + outArr[zi][yi][xi - 2][inDof])/24.0;
        }//end xi
        outArr[zi][yi][xs + nx - 2][outDof] = -((-3.0*outArr[zi][yi][xs + nx - 1][inDof]) - (10.0*outArr[zi][yi][xs + nx - 2][inDof]) +
            (18.0*outArr[zi][yi][xs + nx - 3][inDof]) - (6.0*outArr[zi][yi][xs + nx - 4][inDof])
            + outArr[zi][yi][xs + nx - 5][inDof])/24.0;
        outArr[zi][yi][xs + nx - 1][outDof] = ((25.0*outArr[zi][yi][xs + nx - 1][inDof]) - (48.0*outArr[zi][yi][xs + nx - 2][inDof]) + 
            (36.0*outArr[zi][yi][xs + nx - 3][inDof]) - (16.0*outArr[zi][yi][xs + nx - 4][inDof])
            + (3.0*outArr[zi][yi][xs + nx - 5][inDof]))/24.0;
      }//end yi
    }//end zi
    DMDAVecRestoreArrayDOF(da, in, &inArr);
    DMDAVecRestoreArrayDOF(da, out, &outArr);
  } 
}


