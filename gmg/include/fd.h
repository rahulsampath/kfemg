
#ifndef __FD__
#define __FD__

#include "petsc.h"
#include "petscdmda.h"
#include "petscvec.h"

void applyFD(DM da, int K, Vec in, Vec out);

#endif

