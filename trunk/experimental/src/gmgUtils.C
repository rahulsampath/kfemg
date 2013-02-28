
#include <vector>
#include <iostream>
#include "mpi.h"
#include "gmg/include/gmgUtils.h"
#include "common/include/commonUtils.h"

#ifdef DEBUG
#include <cassert>
#endif

void applyFD1D(MPI_Comm comm, std::vector<PetscInt>& partX, Vec in, Vec out) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  int nx = partX[rank];

  double* inArr;
  double* outArr;
  VecGetArray(in, &inArr);
  VecGetArray(out, &outArr);

  /*
  //First Order
  for(int i = 0; i < (nx - 1); ++i) {
  outArr[i] = (inArr[i + 1] - inArr[i])/2.0;
  }//end i
  outArr[nx - 1] = (inArr[nx - 1] - inArr[nx - 2])/2.0;
  */

  //Second Order
  outArr[0] = -((3.0 * inArr[0]) - (4.0 * inArr[1]) + inArr[2])/4.0;
  for(int i = 1; i < (nx - 1); ++i) {
    outArr[i] = (inArr[i + 1] - inArr[i - 1])/4.0;
  }//end i
  outArr[nx - 1] = ((3.0 * inArr[nx - 1]) - (4.0 * inArr[nx - 2]) + inArr[nx - 3])/4.0;

  /*
  //Fourth Order
  outArr[0] = -((25.0 * inArr[0]) - (48.0 * inArr[1]) + (36.0 * inArr[2]) - (16.0 * inArr[3]) + (3.0 * inArr[4]))/24.0;
  outArr[1] = -((25.0 * inArr[1]) - (48.0 * inArr[2]) + (36.0 * inArr[3]) - (16.0 * inArr[4]) + (3.0 * inArr[5]))/24.0;
  for(int i = 2; i < (nx - 2); ++i) {
  outArr[i] = (-inArr[i + 2] + (8.0 * inArr[i + 1]) - (8.0 * inArr[i - 1]) + inArr[i - 2])/24.0;
  }//end i
  outArr[nx - 2] = ((25.0 * inArr[nx - 2]) - (48.0 * inArr[nx - 3]) + (36.0 * inArr[nx - 4]) - (16.0 * inArr[nx - 5]) + (3.0 * inArr[nx - 6]))/24.0;
  outArr[nx - 1] = ((25.0 * inArr[nx - 1]) - (48.0 * inArr[nx - 2]) + (36.0 * inArr[nx - 3]) - (16.0 * inArr[nx - 4]) + (3.0 * inArr[nx - 5]))/24.0;
  */

  VecRestoreArray(in, &inArr);
  VecRestoreArray(out, &outArr);
}

void computeResidual(Mat mat, Vec sol, Vec rhs, Vec res) {
  //res = rhs - (mat*sol)
  MatMult(mat, sol, res);
  VecAYPX(res, -1.0, rhs);
}

void destroyKSP(std::vector<KSP>& ksp) {
  for(size_t i = 0; i < ksp.size(); ++i) {
    if(ksp[i] != NULL) {
      KSPDestroy(&(ksp[i]));
    }
  }//end i
  ksp.clear();
}

void destroyMat(std::vector<Mat>& mat) {
  for(size_t i = 0; i < mat.size(); ++i) {
    if(mat[i] != NULL) {
      MatDestroy(&(mat[i]));
    }
  }//end i
  mat.clear();
}

void destroyVec(std::vector<Vec>& vec) {
  for(size_t i = 0; i < vec.size(); ++i) {
    if(vec[i] != NULL) {
      VecDestroy(&(vec[i]));
    }
  }//end i
  vec.clear();
}

/*
   void applyFD1D(MPI_Comm comm, std::vector<PetscInt>& partX, Vec in, Vec out) {
   int rank;
   MPI_Comm_rank(comm, &rank);

   int nx = partX[rank];
   int px = partX.size();

   double* inArr;
   VecGetArray(in, &inArr);

   double left1;
   double left2;
   double right1;
   double right2;

   MPI_Request reqR1;
   MPI_Request reqR2;
   MPI_Request reqS1;
   MPI_Request reqS2;

   if(rank < (px - 1)) {
   MPI_Irecv(&right1, 1, MPI_DOUBLE, (rank + 1), 2, comm, &reqR2);
   MPI_Isend(&(inArr[nx - 1]), 1, MPI_DOUBLE, (rank + 1), 1, comm, &reqS1);
   }

   if(rank > 0) {
   MPI_Irecv(&left1, 1, MPI_DOUBLE, (rank - 1), 1, comm, &reqR1);
   MPI_Isend(&(inArr[0]), 1, MPI_DOUBLE, (rank - 1), 2, comm, &reqS2);
   }

   if(rank < (px - 1)) {
   MPI_Status status;
   MPI_Wait(&reqR2, &status);
   }

   MPI_Request reqR3;
   MPI_Request reqS3;

   if(partX[0] == 1) {
   if(rank == 1) {
   if(partX[1] == 1) {
   MPI_Isend(&right1, 1, MPI_DOUBLE, 0, 3, comm, &reqS3);
   } else {
   MPI_Isend(&(in[1]), 1, MPI_DOUBLE, 0, 3, comm, &reqS3);
   }
   }
   if(rank == 0) {
   MPI_Irecv(&right2, 1, MPI_DOUBLE, 1, 3, comm, &reqR3);
   }
   }

   if(rank > 0) {
   MPI_Status status;
   MPI_Wait(&reqR1, &status);
   }

   MPI_Request reqR4;
   MPI_Request reqS4;

   if(partX[px - 1] == 1) {
   if(rank == (px - 2)) {
   if(partX[px - 2] == 1) {
   MPI_Isend(&left1, 1, MPI_DOUBLE, (px - 1), 4, comm, &reqS4);
   } else {
   MPI_Isend(&(in[nx - 2]), 1, MPI_DOUBLE, (px - 1), 4, comm, &reqS4);
   }
   } 
   if(rank == (px - 1)) {
   MPI_Irecv(&left2, 1, MPI_DOUBLE, (px - 2), 4, comm, &reqR4);
   }
}

if(rank < (px - 1)) {
  MPI_Status status;
  MPI_Wait(&reqS1, &status);
}

if(rank > 0) {
  MPI_Status status;
  MPI_Wait(&reqS2, &status);
}

if(partX[0] == 1) {
  if(rank == 1) {
    MPI_Status status;
    MPI_Wait(&reqS3, &status);
  }
  if(rank == 0) {
    MPI_Status status;
    MPI_Wait(&reqR3, &status);
  }
}

if(partX[px - 1] == 1) {
  if(rank == (px - 2)) {
    MPI_Status status;
    MPI_Wait(&reqS4, &status);
  } 
  if(rank == (px - 1)) {
    MPI_Status status;
    MPI_Wait(&reqR4, &status);
  }
}

double* outArr;
VecGetArray(out, &outArr);

for(int i = 1; i < (nx - 1); ++i) {
  outArr[i] = 0.25*(inArr[i + 1] - inArr[i - 1]);
}//end i

if(rank == 0) {
  if(nx == 1) {
    outArr[0] = -0.25*((3.0*inArr[0]) - (4.0*right1) + right2);
  } else if(nx == 2) {
    outArr[0] = -0.25*((3.0*inArr[0]) - (4.0*inArr[1]) + right1);
    outArr[1] = 0.25*(right1 - inArr[0]);
  } else {
    outArr[0] = -0.25*((3.0*inArr[0]) - (4.0*inArr[1]) + inArr[2]);
    if(px == 1) {
      outArr[nx - 1] = 0.25*((3.0*inArr[nx - 1]) - (4.0*inArr[nx - 2]) + inArr[nx - 3]);
    } else {
      outArr[nx - 1] = 0.25*(right1 - inArr[nx - 2]);
    }
  }
} else if(rank == (px - 1)) {
  if(nx == 1) {
    outArr[0] = 0.25*((3.0*inArr[0]) - (4.0*left1) + left2);
  } else if(nx == 2) {
    outArr[0] = 0.25*(inArr[1] - left1);
    outArr[1] = 0.25*((3.0*inArr[1]) - (4.0*inArr[0]) + left1);
  } else {
    outArr[0] = 0.25*(inArr[1] - left1);
    outArr[nx - 1] = 0.25*((3.0*inArr[nx - 1]) - (4.0*inArr[nx - 2]) + inArr[nx - 3]);
  }
} else {
  if(nx == 1) {
    outArr[0] = 0.25*(right1 - left1);
  } else {
    outArr[0] = 0.25*(inArr[1] - left1);
    outArr[nx - 1] = 0.25*(right1 - inArr[nx - 2]);
  }
}

VecRestoreArray(in, &inArr);
VecRestoreArray(out, &outArr);
}
*/


