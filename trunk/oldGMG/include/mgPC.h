
#ifndef __MG_PC__
#define __MG_PC__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscksp.h"
#include "petscpc.h"
#include "gmg/include/smoother.h"
#include <vector>

struct MGdata {
  int K;
  DM daFinest;
  std::vector<Mat> Kmat;
  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec; 
  std::vector<SmootherData*> sData;
  KSP coarseSolver;
  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;
};

PetscErrorCode applyMG(PC pc, Vec in, Vec out);

void applyVcycle(int currLev, MGdata* data);

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

#endif

