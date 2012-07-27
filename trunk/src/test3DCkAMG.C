
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstdlib>

#include "mpi.h"
#include "ml_include.h"
#include "kfemgUtils.h"
#include "gaussQuad.h"
#include "tpShapeFunctions.h"

#define LOC_TO_GLOB(i, j, k, Nx, Ny) (((((k)*(Ny)) + (j))*(Nx)) + (i)) 

struct GlobalData {
  int Nx, Ny, Nz;
  int K;
  std::vector<std::vector<unsigned int> > nonZeroCols;
  std::vector<std::vector<double> > mat;
} myData;

void applyBCs(){

  int dofspNd = pow((myData.K+1),3);

  for(unsigned int i = 0; i < (dofspNd*(myData.Nx *myData.Ny *myData.Nz )); ++i) {
    unsigned int id1, id2;

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

double evaldphi(std::vector<long long int> & coeffs, double xsi, double xeta, double xzeta, 
    std::vector<double> & dsietazetadx, std::vector<double> & dsietazetady, std::vector<double> & dsietazetadz)
{

  // number of shape functions in each direction
  int jsi   = 2*( myData.K + 1);
  //total number of sets including derivatives
  int jdim  = ( myData.K + 2);

  std::vector<double> si(jsi*jdim);
  std::vector<double> eta(jsi*jdim);
  std::vector<double> zeta(jsi*jdim);

  for(int k=0; k< jdim; k++){
    for(int j=0; j< jsi ; ++j){
      // number of coefficients in rational form
      for(int i=0; i< jsi; i++){
        int idx =  k* jsi *jsi * 2 + j * 2 * jsi + 2 * i;   //?????
        si[k*jsi + j] += (coeffs[idx]/coeffs[idx+1])*pow(xsi,idx/jsi);
        eta[k*jsi+ j] += (coeffs[idx]/coeffs[idx+1])*pow(xeta,idx/jsi);
        zeta[k*jsi+j] += (coeffs[idx]/coeffs[idx+1])*pow(xzeta,idx/jsi);
      }
    }
  }

  tensorProduct3D(myData.K, si, eta, zeta, dsietazetadx, 1, 0, 0);
  tensorProduct3D(myData.K, si, eta, zeta, dsietazetady, 0, 1, 0);
  tensorProduct3D(myData.K, si, eta, zeta, dsietazetadz, 0, 0, 1);

}

void computeMatrix() {

  int numNdpEl = 8;
  int dofspNd = pow((myData.K+1),3);
  int dofspEl = dofspNd*numNdpEl ;
  int qtPts = 2*myData.K+2;

  int dofId=0;
  typedef int* intPtr;
  intPtr elemDofMap;
  elemDofMap = new int [dofspNd*numNdpEl];
  for(int j=0;j<dofspNd*numNdpEl;j++){
    elemDofMap[j] = dofId; 
    ++dofId;
  }

  std::vector<double> gaussWts(qtPts);
  std::vector<double> gaussPts(qtPts);
  gaussQuad(gaussPts, gaussWts);  

  double elementMat[dofspNd*numNdpEl][dofspNd*numNdpEl];

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(myData.K, coeffs);

  std::vector<double> dsietazetadx ( pow((myData.K+1),3)*8 );
  std::vector<double> dsietazetady ( pow((myData.K+1),3)*8 );
  std::vector<double> dsietazetadz ( pow((myData.K+1),3)*8 );

  std::vector<double> dNdx_qts (pow((myData.K+1),3)*8 * qtPts * qtPts * qtPts);
  std::vector<double> dNdy_qts (pow((myData.K+1),3)*8 * qtPts * qtPts * qtPts);
  std::vector<double> dNdz_qts (pow((myData.K+1),3)*8 * qtPts * qtPts * qtPts);

  for(int i = 0; i < pow((myData.K+1),3)*8 ; ++i) {
    for(int e = 0; e < qtPts; ++e) {
      for(int f = 0; f < qtPts; ++f) {
        for(int g = 0; g < qtPts; ++g) {
          evaldphi(coeffs, gaussPts[e], gaussPts[f], gaussPts[g], dsietazetadx, dsietazetady, dsietazetadz);
          dNdx_qts.insert(dNdx_qts.end(), dsietazetadx.begin(), dsietazetadx.end());
          dNdy_qts.insert(dNdy_qts.end(), dsietazetady.begin(), dsietazetady.end());
          dNdz_qts.insert(dNdz_qts.end(), dsietazetadz.begin(), dsietazetadz.end());
        }//end g
      }//end f
    }//end e
  }

  int qp3 = qtPts* qtPts* qtPts; 
  int qp2 = qtPts* qtPts; 
  int qp  = qtPts; 

  for(int i = 0; i < dofspNd*numNdpEl; ++i) {
      for(int j = 0; j < dofspNd*numNdpEl; ++j) {
          elementMat[elemDofMap[i]][elemDofMap[j]] = 0.0;
          for(int e = 0; e < qtPts; ++e) {
            for(int f = 0; f < qtPts; ++f) {
              for(int g = 0; g < qtPts; ++g) {

                int idx = i*qp3 + e*qp2 + f*qp + g;
                int jdx = j*qp3 + e*qp2 + f*qp + g;
                
                elementMat[elemDofMap[i]][elemDofMap[j]] += (gaussWts[e]*gaussWts[f]*gaussWts[g]*(
                      (dNdx_qts[idx]*dNdx_qts[jdx]) + (dNdy_qts[idx]*dNdy_qts[jdx]) + (dNdz_qts[idx]*dNdz_qts[jdx]) ));

              }//end g
            }//end f
          }//end e
      }//end for j
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
            int row = (dofspNd*((((kelem + zi[rn])*myData.Ny + (jelem + yi[rn]))*myData.Nx) + ielem + xi[rn])) + rd;
            for(int cn = 0; cn < 8; ++cn) {
              for(int cd = 0; cd < dofspNd; ++cd) {
                int col = (dofspNd*((((kelem + zi[cn])*myData.Ny + (jelem + yi[cn]))*myData.Nx) + ielem + xi[cn])) + cd;
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

  abort();
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

  int dofspNd = pow((myData.K+1),3);
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

  double* solArr = new double[dofspNd*(myData.Nx * myData.Ny * myData.Nz)];
  double* rhsArr = new double[dofspNd*(myData.Nx * myData.Ny * myData.Nz)];

  for(int i = 0; i < (dofspNd*(myData.Nx * myData.Ny * myData.Nz)); i++) {
    solArr[i] = (static_cast<double>(rand()))/(static_cast<double>(RAND_MAX));
  }//end for i

  myMatVec(NULL, (dofspNd*(myData.Nx * myData.Ny * myData.Nz)), solArr, (dofspNd*(myData.Nx * myData.Ny * myData.Nz)), rhsArr);

  for(int i = 0; i < (dofspNd*(myData.Nx * myData.Ny * myData.Nz)); i++) {
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



