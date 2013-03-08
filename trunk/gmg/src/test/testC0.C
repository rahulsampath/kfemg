
#include "petsc.h"
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/mesh.h"
#include "gmg/include/mms.h"
#include "gmg/include/mgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/assembly.h"
#include "gmg/include/intergrid.h"
#include "common/include/commonUtils.h"
#include <iomanip>
#include <iostream>

#ifdef DEBUG
#include <cassert>
#endif

PetscClassId gmgCookie;
PetscLogEvent meshEvent;
PetscLogEvent buildPmatEvent;
PetscLogEvent buildKmatEvent;
PetscLogEvent rhsEvent;
PetscLogEvent solverSetupEvent;
PetscLogEvent solverApplyEvent;
PetscLogEvent errEvent;
PetscLogEvent cleanupEvent;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  PETSC_COMM_WORLD = MPI_COMM_WORLD;

  PetscInitialize(&argc, &argv, "optionsC0", PETSC_NULL);

  PetscClassIdRegister("GMG", &gmgCookie);
  PetscLogEventRegister("Mesh", gmgCookie, &meshEvent);
  PetscLogEventRegister("BuildPmat", gmgCookie, &buildPmatEvent);
  PetscLogEventRegister("BuildKmat", gmgCookie, &buildKmatEvent);
  PetscLogEventRegister("RHS", gmgCookie, &rhsEvent);
  PetscLogEventRegister("SolverSetup", gmgCookie, &solverSetupEvent);
  PetscLogEventRegister("SolverApply", gmgCookie, &solverApplyEvent);
  PetscLogEventRegister("Error", gmgCookie, &errEvent);
  PetscLogEventRegister("Cleanup", gmgCookie, &cleanupEvent);


  PetscFinalize();

  destroyComms(activeComms);

  MPI_Finalize();

  return 0;
}



