
#ifndef __FD__
#define __FD__

#include "petsc.h"
#include "petscdmda.h"
#include "petscvec.h"

applyFD(DM da, int K, Vec in, Vec out);

#endif

