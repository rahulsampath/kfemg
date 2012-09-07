
#ifndef __GMG_UTILS__
#define __GMG_UTILS__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscda.h"
#include "petscksp.h"
#include "petscpc.h"
#include <vector>

bool foundValidDApart(int dim, PetscInt Nz, PetscInt Ny, PetscInt Nx, int npes);

void createGridSizes(int dim, std::vector<PetscInt> & Nz, std::vector<PetscInt> & Ny, std::vector<PetscInt> & Nx);

#endif

