
#ifndef __MG_PC__
#define __MG_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscdmda.h"
#include "petscmat.h"
#include "petscksp.h"
#include "petscpc.h"
#include "gmg/include/static/smoother.h"
#include <vector>


void setupMG(PC pc, int K, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<long long int> >& coeffs, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec);


#endif

