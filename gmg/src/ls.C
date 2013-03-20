
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

void applyLS(LSdata* data, Vec g, Vec v1, Vec v2, double c[2],
    int maxIters, double tgtNorm, double currNorm) {
  MatMult((data->Kmat), v1, (data->w1));
  MatMult((data->Kmat), v2, (data->w2));
  double Hmat[2][2];
  VecDot((data->w1), (data->w1), &(Hmat[0][0]));
  VecDot((data->w1), (data->w2), &(Hmat[0][1]));
  Hmat[1][0] = Hmat[0][1];
  VecDot((data->w2), (data->w2), &(Hmat[1][1]));
  double Hinv[2][2];
  matInvert2x2(Hmat, Hinv);
  double w1g;
  double w2g;
  VecDot((data->w1), g, &w1g);
  VecDot((data->w2), g, &w2g);
  c[0] = c[1] = 0.0;
  double gDotG = currNorm*currNorm; 
  double tgtNrmSqr = tgtNorm*tgtNorm;
  double obj = gDotG;
  for(int iter = 0; iter < maxIters; ++iter) {
    if(obj <= tgtNrmSqr) {
      break;
    }
    double grad[2];
    grad[0] = -w1g + (c[0]*Hmat[0][0]) + (c[1]*Hmat[1][0]);
    grad[1] = -w2g + (c[0]*Hmat[0][1]) + (c[1]*Hmat[1][1]);
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
      tmp[0] = c[0] - (alpha*step[0]);
      tmp[1] = c[1] - (alpha*step[1]);
      double tmpObj = gDotG -2.0*((tmp[0]*w1g) + (tmp[1]*w2g));
      for(int r = 0; r < 2; ++r) {
        for(int c = 0; c < 2; ++c) {
          tmpObj += (tmp[r]*tmp[c]*Hmat[r][c]);
        }//end c
      }//end r
      if(tmpObj < obj) {
        obj = tmpObj;
        c[0] = tmp[0];
        c[1] = tmp[1];
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



