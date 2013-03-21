
#include "gmg/include/ls.h"
#include "common/include/commonUtils.h"

void setupLS(LSdata* data, Mat Kmat) {
  data->Kmat = Kmat;
  MatGetVecs(Kmat, PETSC_NULL, &(data->w1));
  VecDuplicate((data->w1), &(data->w2));  
}

void destroyLS(LSdata* data) {
  VecDestroy(&(data->w1));
  VecDestroy(&(data->w2));
  delete data;
}

double applyLS(LSdata* data, Vec g, Vec v1, Vec v2, double a[2]) {
  MatMult((data->Kmat), v1, (data->w1));
  MatMult((data->Kmat), v2, (data->w2));
  double mat[2][2];
  VecDot((data->w1), (data->w1), &(mat[0][0]));
  VecDot((data->w1), (data->w2), &(mat[0][1]));
  mat[1][0] = mat[0][1];
  VecDot((data->w2), (data->w2), &(mat[1][1]));
  double inv[2][2];
  matInvert2x2(mat, inv);
  double rhs[2];
  VecDot((data->w1), g, &(rhs[0]));
  VecDot((data->w2), g, &(rhs[1]));
  matMult2x2(inv, rhs, a);
  VecAXPBYPCZ(g, (-a[0]), (-a[1]), 1.0, (data->w1), (data->w2));
  double normUpdate = (a[0]*a[0]*mat[0][0]) + (2.0*a[0]*a[1]*mat[0][1]) +
    (a[1]*a[1]*mat[1][1]) - (2.0*((a[0]*rhs[0]) + (a[1]*rhs[1]))); 
  return normUpdate;
}

