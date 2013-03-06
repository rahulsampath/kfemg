
#include "gmg/include/gmgUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res) {
  //res = rhs - (mat*sol)
  MatMult(mat, sol, res);
  VecAYPX(res, -1.0, rhs);
}

void destroyKSP(std::vector<KSP>& ksp) {
  for(size_t i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(&(ksp[i]));
    }
  }//end i
  ksp.clear();
}

void destroyMat(std::vector<Mat>& mat) {
  for(size_t i = 0; i < mat.size(); ++i) {
    if(mat[i] != NULL) {
      MatDestroy(&(mat[i]));
    }
  }//end i
  mat.clear();
}

void destroyVec(std::vector<Vec>& vec) {
  for(size_t i = 0; i < vec.size(); ++i) {
    if(vec[i] != NULL) {
      VecDestroy(&(vec[i]));
    }
  }//end i
  vec.clear();
}



