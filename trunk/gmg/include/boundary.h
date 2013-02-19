
#ifndef __BOUNDARY__
#define __BOUNDARY__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include <vector>

void correctKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, int K);

void dirichletMatrixCorrection(Mat Kmat, DM da, const int K);

void makeBoundariesConsistent(DM da, Vec in, Vec out, const int K);

#endif

