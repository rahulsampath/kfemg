
#ifndef __NEW_RTG_PC__
#define __NEW_RTG_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscksp.h"
#include "petscpc.h"
#include "gmg/include/newSmoother.h"
#include <vector>

void setupNewRTG(PC pc, int currK, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<Mat> >& Kmat, std::vector<std::vector<Mat> >& Pmat, 
    std::vector<std::vector<Vec> >& tmpCvec);

#endif


