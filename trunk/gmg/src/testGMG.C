
#include "mpi.h"
#include "petsc.h"
#include "common/include/commonUtils.h"
#include "gmg/include/gmgUtils.h"

int main(int argc, char *argv[]) {
  PetscInitialize(&argc, &argv, "options", PETSC_NULL);
  
  PetscFinalize();

  return 0;
}


