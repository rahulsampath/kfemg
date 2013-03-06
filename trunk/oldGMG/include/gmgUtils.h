
#ifndef __GMG_UTILS__
#define __GMG_UTILS__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include <vector>

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res);

void destroyMat(std::vector<Mat>& mat);

void destroyVec(std::vector<Vec>& vec);

void destroyKSP(std::vector<KSP>& ksp);

#endif

