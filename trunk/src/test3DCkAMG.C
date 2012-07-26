
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstdlib>
#include "mpi.h"
#include "ml_include.h"
#include "kfemgUtils.h"

#define LOC_TO_GLOB(i, j, k, Nx, Ny) (((((k)*(Ny)) + (j))*(Nx)) + (i)) 

struct GlobalData {
  int Nx, Ny, Nz;
  int K;
  std::vector<std::vector<unsigned int> > nonZeroCols;
  std::vector<std::vector<double> > mat;
} myData;

void applyBCs(){
  for(unsigned int i = 0; i < (dofspNd*(myData.Nx *myData.Ny *myData.Nz )); ++i) {
    unsigned int id1, id2;
    int dofspNd = pow((myData.K+1),3);

    id1 = i; 
    // yz plane
    for(int idz = 0 ; idz < myData.Nz; ++idz) {
      for(int idy = 0; idy < myData.Ny; ++idy) {
        int idx = 0 ;
        id2 = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));

        //surface @ x = 0 
        //row correction  
        for(size_t j = 0; j < myData.nonZeroCols[id1].size(); ++j) {
          if(myData.nonZeroCols[id1][j] == id2) {
            myData.mat[id1][j] = 0.0;
            break;
          }
        }//end for j

        //column correction  
        for(size_t j = 0; j < myData.nonZeroCols[id2].size(); ++j) {
          if(myData.nonZeroCols[id2][j] == id1) {
            myData.mat[id2][j] = 0.0;
            break;
          }
        }//end for j

        idx = myData.Nx - 1;
        id2 = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));

        //surface @ x = L 
        //row correction  
        for(size_t j = 0; j < myData.nonZeroCols[id1].size(); ++j) {
          if(myData.nonZeroCols[id1][j] == id2) {
            myData.mat[id1][j] = 0.0;
            break;
          }
        }//end for j

        //column correction  
        for(size_t j = 0; j < myData.nonZeroCols[id2].size(); ++j) {
          if(myData.nonZeroCols[id2][j] == id1) {
            myData.mat[id2][j] = 0.0;
            break;
          }
        }//end for j

      }//end idy
    }//end idz

    //xz plane
    for(int idz = 0 ; idz < myData.Nz; ++idz) {
      for(int idx = 0; idx < myData.Nx; ++idx) {
        int idy = 0 ;
        id2 = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));

        //surface @ y = 0 
        //row correction  
        for(size_t j = 0; j < myData.nonZeroCols[id1].size(); ++j) {
          if(myData.nonZeroCols[id1][j] == id2) {
            myData.mat[id1][j] = 0.0;
            break;
          }
        }//end for j

        //column correction  
        for(size_t j = 0; j < myData.nonZeroCols[id2].size(); ++j) {
          if(myData.nonZeroCols[id2][j] == id1) {
            myData.mat[id2][j] = 0.0;
            break;
          }
        }//end for j

        idy = myData.Ny - 1;
        id2 = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));

        //surface @ y = L 
        //row correction  
        for(size_t j = 0; j < myData.nonZeroCols[id1].size(); ++j) {
          if(myData.nonZeroCols[id1][j] == id2) {
            myData.mat[id1][j] = 0.0;
            break;
          }
        }//end for j

        //column correction  
        for(size_t j = 0; j < myData.nonZeroCols[id2].size(); ++j) {
          if(myData.nonZeroCols[id2][j] == id1) {
            myData.mat[id2][j] = 0.0;
            break;
          }
        }//end for j

      }//end idx
    }//end idz

    //xy plane
    for(int idy = 0 ; idy < myData.Ny; ++idy) {
      for(int idx = 0; idx < myData.Nx; ++idx) {
        int idz = 0 ;
        id2 = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));

        //surface @ z = 0 
        //row correction  
        for(size_t j = 0; j < myData.nonZeroCols[id1].size(); ++j) {
          if(myData.nonZeroCols[id1][j] == id2) {
            myData.mat[id1][j] = 0.0;
            break;
          }
        }//end for j

        //column correction  
        for(size_t j = 0; j < myData.nonZeroCols[id2].size(); ++j) {
          if(myData.nonZeroCols[id2][j] == id1) {
            myData.mat[id2][j] = 0.0;
            break;
          }
        }//end for j

        idz = myData.Nz - 1;
        id2 = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));

        //surface @ z = L 
        //row correction  
        for(size_t j = 0; j < myData.nonZeroCols[id1].size(); ++j) {
          if(myData.nonZeroCols[id1][j] == id2) {
            myData.mat[id1][j] = 0.0;
            break;
          }
        }//end for j

        //column correction  
        for(size_t j = 0; j < myData.nonZeroCols[id2].size(); ++j) {
          if(myData.nonZeroCols[id2][j] == id1) {
            myData.mat[id2][j] = 0.0;
            break;
          }
        }//end for j

      }//end idx
    }//end idy

  }//end i

  {
    // yz plane
    for(int idz = 0 ; idz < myData.Nz; ++idz) {
      for(int idy = 0; idy < myData.Ny; ++idy) {
        int idx = 0 ;
        unsigned int id = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));
        for(size_t j = 0; j < myData.nonZeroCols[id].size(); ++j) {
          if(myData.nonZeroCols[id][j] == id) {
            myData.mat[id][j] = 1.0;
            break;
          }
        }//end for j
        idx = myData.Nx - 1;
        id = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));
        for(size_t j = 0; j < myData.nonZeroCols[id].size(); ++j) {
          if(myData.nonZeroCols[id][j] == id) {
            myData.mat[id][j] = 1.0;
            break;
          }
        }//end for j
      }//end for idy
    }//end for idz
  }
  {
    // xz plane
    for(int idz = 0 ; idz < myData.Nz; ++idz) {
      for(int idx = 0; idx < myData.Nx; ++idx) {
        int idy = 0 ;
        unsigned int id = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));
        for(size_t j = 0; j < myData.nonZeroCols[id].size(); ++j) {
          if(myData.nonZeroCols[id][j] == id) {
            myData.mat[id][j] = 1.0;
            break;
          }
        }//end for j
        idy = myData.Ny - 1;
        id = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));
        for(size_t j = 0; j < myData.nonZeroCols[id].size(); ++j) {
          if(myData.nonZeroCols[id][j] == id) {
            myData.mat[id][j] = 1.0;
            break;
          }
        }//end for j
      }//end for idx
    }//end for idz
  }
  {
    // xy plane
    for(int idy = 0 ; idy < myData.Ny; ++idy) {
      for(int idx = 0; idx < myData.Nx; ++idx) {
        int idz = 0 ;
        unsigned int id = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));
        for(size_t j = 0; j < myData.nonZeroCols[id].size(); ++j) {
          if(myData.nonZeroCols[id][j] == id) {
            myData.mat[id][j] = 1.0;
            break;
          }
        }//end for j
        idz = myData.Nz - 1;
        id = dofspNd*(LOC_TO_GLOB(idx,idy,idz,myData.Nx,myData.Ny));
        for(size_t j = 0; j < myData.nonZeroCols[id].size(); ++j) {
          if(myData.nonZeroCols[id][j] == id) {
            myData.mat[id][j] = 1.0;
            break;
          }
        }//end for j
      }//end for idx
    }//end for idy
  }

}

double evalPhiPrime(int dofID, double xsi, double xeta, double xzeta){

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(myData.K, coeffs);

  for(int j; j< ; ++j){
    for(int i=0; i< ; ++i){
      si[j] += (coeffs[i]/coeffs[i+1])*pow(xsi,i);
    }
    for(int i=0; i< ; ++i){
      eta[j] += (coeffs[i]/coeffs[i+1])*pow(xeta,i);
    }
    for(int i=0; i< ; ++i){
      zeta[j] += (coeffs[i]/coeffs[i+1])*pow(xzeta,i);
    }
  }

  tensorProduct3D(si, eta, zeta, sietazeta);
  assert(coeffs[0] == 1LL);

}

void computeMatrix() {

  int numNdpEl = 8;
  int dofspNd = pow((myData.K+1),3);
  int dofspEl = dofspNd*numNdpEl ;
  int qtPts = 2*myData.K+1;

  int dofId=0;
  typedef int** int2Ptr;
  typedef int* intPtr;
  int2Ptr elemDofMap = new int2Ptr[numNdpEl];
  for(int i=0;i<numNdpEl;i++){
    elemDofMap[i] = new double [dofspNd];
    for(int j=0;j<dofspNd;j++){
      elemDofMap[i][j] = dofId; 
      ++dofId;
    }
  }

  double elementMat[dofspNd*numNdpEl][dofspNd*numNdpEl];
  for(int i = 0; i < numNdpEl; ++i) {
    for(int c = 0; c < dofspNd; ++c) {
      for(int j = 0; j < numNdpEl; ++j) {
        for(int d = 0; d < dofspNd; ++d) {
          elementMat[elemDofMap[i][c]][elemDofMap[j][d]] = 0.0;
          for(int e = 0; e < qtPts; ++e) {
            for(int f = 0; f < qtPts; ++f) {
              for(int g = 0; g < qtPts; ++g) {
                elementMat[elemDofMap[i][c]][elemDofMap[j][d]] += (gaussWts[e]*gaussWts[f]*gaussWts[g]*(
                      (evalPhiPrime(elemDofMap[i][c], gaussPts[e], gaussPts[f], gaussPts[g])*evalPhiPrime(elemDofMap[j][d], gaussPts[e], gaussPts[f], gaussPts[g])) + 
                      (evalPhiPrime(elemDofMap[i][c], gaussPts[e], gaussPts[f], gaussPts[g])*evalPhiPrime(elemDofMap[j][d], gaussPts[e], gaussPts[f], gaussPts[g])) + 
                      (evalPhiPrime(elemDofMap[i][c], gaussPts[e], gaussPts[f], gaussPts[g])*evalPhiPrime(elemDofMap[j][d], gaussPts[e], gaussPts[f], gaussPts[g]))  
                      ));
              }//end g
            }//end f
          }//end e
        }//end for d
      }//end for j
    }//end for c
  }//end for i

  myData.nonZeroCols.resize(dofspNd*(myData.Nx * myData.Ny * myData.Nz));
  myData.mat.resize(dofspNd*(myData.Nx * myData.Ny * myData.Nz));

  for(int kelem = 0; kelem < (myData.Nz - 1); ++kelem) {
    for(int jelem = 0; jelem < (myData.Ny - 1); ++jelem) {
      for(int ielem = 0; ielem < (myData.Nx - 1); ++ielem) {
        int xi[] = {0, 1, 0, 1, 0, 1, 0,1};
        int yi[] = {0, 0, 1, 1, 0, 0, 1,1};
        int zi[] = {0, 0, 0, 0, 1, 1, 1,1};
        for(int rn = 0; rn < 8; ++rn) {
          for(int rd = 0; rd < dofspNd; ++rd) {
            int row = (dofspNd*((((kelem + zi[rn])*Ny + (jelem + yi[rn]))*Nx) + ielem + xi[rn])) + rd;
            for(int cn = 0; cn < 8; ++cn) {
              for(int cd = 0; cd < dofspNd; ++cd) {
                int col = (dofspNd*((((kelem + zi[cn])*Ny + (jelem + yi[cn]))*Nx) + ielem + xi[cn])) + cd;
                int idx = -1;
                for(size_t k = 0; k < myData.nonZeroCols[row].size(); ++k) {
                  if(myData.nonZeroCols[row][k] == col) {
                    idx = k;
                    break;
                  }
                }//end for k
                if(idx == -1) {
                  (myData.nonZeroCols[row]).push_back(col);
                  (myData.mat[row]).push_back(elementMat[(dofspNd*rn) + rd][(dofspNd*cn) + cd]);
                } else {
                  myData.mat[row][idx] += (elementMat[(dofspNd*rn) + rd][(dofspNd*cn) + cd]);
                }
              }//end cd
            }//end cn
          }//end rd
        }//end rn
      }//end for ielem
    }//end for jelem
  }//end for kelem

  applyBCs();
}



int myMatVec(ML_Operator *data, int in_length, double in[], int out_length, double out[]) {
  for(int i = 0; i < out_length; ++i) {
    out[i] = 0.0;
    for(size_t j = 0; j < (myData.nonZeroCols[i]).size(); ++j) {
      out[i] += ((myData.mat[i][j])*in[myData.nonZeroCols[i][j]]);
    }//end for j
  }//end for i
  return 0;
}

int myGetRow(ML_Operator *data, int N_requested_rows, int requested_rows[],
    int allocated_space, int columns[], double values[], int row_lengths[]) {
  int spaceRequired = 0;
  int cnt = 0;
  for(int i = 0; i < N_requested_rows; ++i) {
    int row = requested_rows[i];
    std::vector<unsigned int> cols = myData.nonZeroCols[row];
    std::vector<double> vals = myData.mat[row];

    spaceRequired += cols.size();
    if(allocated_space >= spaceRequired) {
      for(size_t j = 0; j < cols.size(); ++j) {
        columns[cnt] = cols[j];
        values[cnt] = vals[j];
        ++cnt;
      }//end for j
      row_lengths[i] = cols.size();
    } else {
      return 0;
    }
  }//end for i
  return 1;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  assert(argc > 2);
  const int numGrids = atoi(argv[1]);
  myData.Nx = atoi(argv[2]);
  myData.Ny = atoi(argv[2]);
  myData.Nz = atoi(argv[2]);
  myData.K = atoi(argv[3]);

  double computeMatStartTime = MPI_Wtime();
  computeMatrix();
  double computeMatEndTime = MPI_Wtime();

  std::cout<<"Mat create time = "<<(computeMatEndTime - computeMatStartTime)<<std::endl;

  const int numPDEs = pow((myData.K+1),3);
  int maxIterations = 1000;
  if(numGrids == 1) {
    maxIterations = 1;
  }
  const int coarseSize = 8;

  double setupStart = MPI_Wtime();
  ML_set_random_seed(123456);
  ML* ml_object;
  ML_Create(&ml_object, numGrids);

  ML_Init_Amatrix(ml_object, 0, dofspNd*(myData.Nx * myData.Ny * myData.Nz), dofspNd*(myData.Nx * myData.Ny * myData.Nz), &myData);
  ML_Set_Amatrix_Getrow(ml_object, 0, &myGetRow, NULL, dofspNd*(myData.Nx * myData.Ny * myData.Nz));
  ML_Set_Amatrix_Matvec(ml_object, 0, &myMatVec);
  ML_Set_MaxIterations(ml_object, maxIterations);
  ML_Set_Tolerance(ml_object, 1.0e-12);
  ML_Set_ResidualOutputFrequency(ml_object, 1);
  ML_Set_PrintLevel(10);
  ML_Set_OutputLevel(ml_object, 10);

  ML_Aggregate* agg_object;
  ML_Aggregate_Create(&agg_object);
  agg_object->num_PDE_eqns = numPDEs;
  agg_object->nullspace_dim = 1;
  ML_Aggregate_Set_MaxCoarseSize(agg_object, coarseSize);
  ML_Aggregate_Set_CoarsenScheme_UncoupledMIS(agg_object);

  const int nlevels = ML_Gen_MGHierarchy_UsingAggregation(ml_object, 0, ML_INCREASING, agg_object);
  std::cout<<"Number of actual levels: "<<nlevels<<std::endl;

  for(int lev = 0; lev < (nlevels - 1); ++lev) {
    ML_Gen_Smoother_SymGaussSeidel(ml_object, lev, ML_BOTH, 2, 1.0);
    //ML_Gen_Smoother_Jacobi(ml_object, lev, ML_BOTH, 2, 0.8);
  }
  ML_Gen_Smoother_Amesos(ml_object, (nlevels - 1), ML_AMESOS_KLU, -1, 0.0);

  ML_Gen_Solver(ml_object, ML_MGV, 0, (nlevels-1));

  double setupEnd = MPI_Wtime();

  double* solArr = new double[2*(myData.N)];
  double* rhsArr = new double[2*(myData.N)];

  for(int i = 0; i < (2*(myData.N)); i++) {
    solArr[i] = (static_cast<double>(rand()))/(static_cast<double>(RAND_MAX));
  }//end for i

  myMatVec(NULL, (2*myData.N), solArr, (2*myData.N), rhsArr);

  for(int i = 0; i < (2*(myData.N)); i++) {
    solArr[i] = 0.0;
  }//end for i

  double solveStart = MPI_Wtime();

  ML_Iterate(ml_object, solArr, rhsArr);

  double solveEnd = MPI_Wtime();

  ML_Aggregate_Destroy(&agg_object);
  ML_Destroy(&ml_object);

  std::cout<<"Setup Time = "<<(setupEnd - setupStart)<<std::endl;
  std::cout<<"Solve Time = "<<(solveEnd - solveStart)<<std::endl;

  delete [] solArr;
  delete [] rhsArr;
  MPI_Finalize();
}  



