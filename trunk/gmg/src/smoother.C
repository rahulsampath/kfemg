
#include "gmg/include/smoother.h"
#include "gmg/include/gmgUtils.h"
#include <iostream>

#include <cassert>
//#include <iomanip>

void setupSmootherData(SmootherData* data, Mat Kmat) {
  MPI_Comm comm;
  PetscObjectGetComm(((PetscObject)Kmat), &comm);
  data->maxIts = 4;
  PetscOptionsGetInt(PETSC_NULL, "-smootherMaxIts", &(data->maxIts), PETSC_NULL);
  data->tol = 0.5;
  PetscOptionsGetReal(PETSC_NULL, "-smootherTol", &(data->tol), PETSC_NULL);
  data->Kmat = Kmat;
  KSPCreate(comm, &(data->ksp1));
  KSPCreate(comm, &(data->ksp2));
  KSPSetType(data->ksp1, KSPCG);
  KSPSetType(data->ksp2, KSPGMRES);
  KSPSetPCSide(data->ksp1, PC_LEFT);
  KSPSetPCSide(data->ksp2, PC_RIGHT);
  PC pc1;
  PC pc2;
  KSPGetPC(data->ksp1, &pc1);
  KSPGetPC(data->ksp2, &pc2);
  PCSetType(pc1, PCNONE);
  PCSetType(pc2, PCNONE);
  KSPSetInitialGuessNonzero(data->ksp1, PETSC_TRUE);
  KSPSetInitialGuessNonzero(data->ksp2, PETSC_TRUE);
  KSPSetOperators(data->ksp1, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetOperators(data->ksp2, Kmat, Kmat, SAME_PRECONDITIONER);
  KSPSetTolerances(data->ksp1, 1.0e-12, 1.0e-12, 2.0, 2);
  KSPSetTolerances(data->ksp2, 1.0e-12, 1.0e-12, 2.0, 2);
  KSPDefaultConvergedSetUIRNorm(data->ksp1);
  KSPDefaultConvergedSetUIRNorm(data->ksp2);
  KSPSetNormType(data->ksp1, KSP_NORM_UNPRECONDITIONED);
  KSPSetNormType(data->ksp2, KSP_NORM_UNPRECONDITIONED);
  KSPSetOptionsPrefix(data->ksp1, "smooth1_");
  KSPSetOptionsPrefix(data->ksp2, "smooth2_");
  KSPSetFromOptions(data->ksp1);
  KSPSetFromOptions(data->ksp2);
  MatGetVecs(Kmat, PETSC_NULL, &(data->res));
}

void applySmoother(SmootherData* data, Vec in, Vec out) {
  computeResidual(data->Kmat, out, in, data->res);
  PetscReal resNorm;
  VecNorm(data->res, NORM_2, &resNorm);
  PetscReal initNorm = resNorm;
  //std::cout<<"InitNorm in smoother = "<<std::setprecision(13)<<initNorm<<std::endl;
  //std::cout<<"New tol = "<<std::setprecision(13)<<newTol<<std::endl;
  bool done = false;
  for(int iter = 0; iter < (data->maxIts); ++iter) {
    for(int subIt = 0; subIt < 2; ++subIt) {
      //std::cout<<"Smooth iter = "<<iter<<" sub = "<<subIt<<" res = "<<std::setprecision(13)<<resNorm<<std::endl;
      if(resNorm < 1.0e-12) {
        done = true;
        break;
      }
      if(resNorm < (initNorm*(data->tol))) {
        done = true;
        break;
      }
      PetscReal newTol = initNorm*(data->tol)/resNorm;
      if(newTol >= 1.0) {
        newTol = 0.999999;
      }
      if(subIt == 0) {
        KSPSetTolerances(data->ksp1, newTol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        KSPSolve(data->ksp1, in, out);
      } else {
        KSPSetTolerances(data->ksp2, newTol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        KSPSolve(data->ksp2, in, out);
      }
      computeResidual(data->Kmat, out, in, data->res);
      VecNorm(data->res, NORM_2, &resNorm);
    }//end subIt
    if(done) {
      break;
    }
  }//end iter
  if((resNorm >= 1.0e-12) && (resNorm >= (initNorm*(data->tol)))) {
    assert(false);
  }
}

void destroySmootherData(SmootherData* data) {
  KSPDestroy(&(data->ksp1));
  KSPDestroy(&(data->ksp2));
  VecDestroy(&(data->res));
  delete data;
}


