
#ifndef __SMOOTHER__
#define __SMOOTHER__

#include "petsc.h"
#include "petscvec.h"

struct SmootherData {
};

void setupSmootherData(SmootherData* data);

void destroySmootherData(SmootherData* data);

void applySmoother(SmootherData* data, Vec in, Vec out);

#endif

