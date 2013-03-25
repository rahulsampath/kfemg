
#include "gmg/include/loa.h"

void setupLOA(LOAdata* data, int K, DM daL, DM daH,
    std::vector<std::vector<long long int> >& coeffs) {
  data->K = K;
  data->coeffs = &coeffs;
  data->daL = daL;
  data->daH = daH;
}

void destroyLOA(LOAdata* data) {
  delete data;
}

void applyLOA(LOAdata* data, Vec high, Vec low) {
  PetscInt dim;
  PetscInt Nx;
  PetscInt Ny;
  PetscInt Nz;
  PetscInt dofsPerNode;
  DMDAGetInfo(data->daH, &dim, &Nx, &Ny, &Nz, PETSC_NULL, PETSC_NULL, PETSC_NULL,
      &dofsPerNode, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  PetscInt xs;
  PetscInt ys;
  PetscInt zs;
  PetscInt nx;
  PetscInt ny;
  PetscInt nz;
  DMDAGetCorners(data->daH, &xs, &ys, &zs, &nx, &ny, &nz);

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
    PetscScalar** arr;
    DMDAVecGetArrayDOF(data->daH, high, &arr);
    for(int xi = xs; xi < (xs + nx); ++xi) {
      for(int d = 0; d < dofsPerNode; ++d) {
      }//end d
    }//end xi
    DMDAVecRestoreArrayDOF(data->daH, high, &arr);
  } else if(dim == 2) {
    PetscScalar*** arr;
    DMDAVecGetArrayDOF(data->daH, high, &arr);
    DMDAVecRestoreArrayDOF(data->daH, high, &arr);
  } else {
    PetscScalar**** arr;
    DMDAVecGetArrayDOF(data->daH, high, &arr);
    DMDAVecRestoreArrayDOF(data->daH, high, &arr);
  }

}

/*
   void applyLS(LSdata* data, Vec g, Vec v1, Vec v2, double a[2],
   int maxIters, double tgtNorm, double currNorm) {
   MatMult((data->Kmat), v1, (data->w1));
   MatMult((data->Kmat), v2, (data->w2));
   double Hmat[2][2];
   VecDot((data->w1), (data->w1), &(Hmat[0][0]));
   VecDot((data->w1), (data->w2), &(Hmat[0][1]));
   Hmat[1][0] = Hmat[0][1];
   VecDot((data->w2), (data->w2), &(Hmat[1][1]));
   double eig[2];
   eigenVals2x2(Hmat, eig);
   double minEig = ((eig[0] < eig[1]) ? (eig[0]) : (eig[1]));
   double Hinv[2][2];
   if(minEig <= 1.0e-12) {
   double shift = 1.0 - minEig;
   double Lmat[2][2];
   Lmat[0][0] = Hmat[0][0] + shift;
   Lmat[0][1] = Hmat[0][1];
   Lmat[1][0] = Hmat[1][0];
   Lmat[1][1] = Hmat[1][1] + shift;
   matInvert2x2(Lmat, Hinv);
   } else {
   matInvert2x2(Hmat, Hinv);
   }
   double w1g;
   double w2g;
   VecDot((data->w1), g, &w1g);
   VecDot((data->w2), g, &w2g);
   a[0] = a[1] = 0.0;
   double gDotG = currNorm*currNorm; 
   double tgtNrmSqr = tgtNorm*tgtNorm;
   double obj = gDotG;
   for(int iter = 0; iter < maxIters; ++iter) {
   if(obj <= tgtNrmSqr) {
   break;
   }
   double grad[2];
   grad[0] = -w1g + (a[0]*Hmat[0][0]) + (a[1]*Hmat[1][0]);
   grad[1] = -w2g + (a[0]*Hmat[0][1]) + (a[1]*Hmat[1][1]);
   if((fabs(grad[0]) <= 1.0e-12) && (fabs(grad[1]) <= 1.0e-12)) {
   break;
   }
   double step[2];
   matMult2x2(Hinv, grad, step);
   if((fabs(step[0]) <= 1.0e-12) && (fabs(step[1]) <= 1.0e-12)) {
   break;
   }
   double alpha = 1.0;
   while(alpha >= 1.0e-12) {
   double tmp[2]; 
   tmp[0] = a[0] - (alpha*step[0]);
   tmp[1] = a[1] - (alpha*step[1]);
   double tmpObj = gDotG -2.0*((tmp[0]*w1g) + (tmp[1]*w2g));
   for(int r = 0; r < 2; ++r) {
   for(int c = 0; c < 2; ++c) {
   tmpObj += (tmp[r]*tmp[c]*Hmat[r][c]);
   }//end c
   }//end r
   if(tmpObj < obj) {
   obj = tmpObj;
   a[0] = tmp[0];
   a[1] = tmp[1];
   break;
   } else {
   alpha *= 0.5;
   }
   }
   if(alpha < 1.0e-12) {
   break;
   }
}//end iter
}
*/


