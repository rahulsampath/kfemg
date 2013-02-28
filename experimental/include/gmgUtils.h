
#ifndef __GMG_UTILS__
#define __GMG_UTILS__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscdmda.h"
#include "petscksp.h"
#include "petscpc.h"
#include <vector>
#include "mpi.h"
#include "common/include/commonUtils.h"

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res);

void applyFD1D(MPI_Comm comm, std::vector<PetscInt>& partX, Vec in, Vec out);

void destroyMat(std::vector<Mat>& mat);

void destroyVec(std::vector<Vec>& vec);

void destroyKSP(std::vector<KSP>& ksp);

#endif



