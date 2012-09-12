
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "ml_include.h"
#include "amg/include/amgUtils.h"
#include "common/include/commonUtils.h"

void zeroBoundaries(double* arr, const unsigned int K, const unsigned int dim,
    const int Nz, const int Ny, const int Nx) {
  if(dim > 1) {
    assert(Ny > 1);
  } else {
    assert(Ny == 1);
  }
  if(dim > 2) {
    assert(Nz > 1);
  } else {
    assert(Nz == 1);
  }
  assert(dim > 0);
  assert(Nx > 1);

  int dofsPerNode = getDofsPerNode(dim, K); 

  std::vector<int> xVec; 
  xVec.push_back(0);
  xVec.push_back((Nx - 1));

  std::vector<int> yVec; 
  if(dim > 1) {
    yVec.push_back(0);
    yVec.push_back((Ny - 1));
  }

  std::vector<int> zVec;
  if(dim > 2) {
    zVec.push_back(0);
    zVec.push_back((Nz - 1));
  }

  //x
  for(int c = 0; c < xVec.size(); ++c) {
    int xi = xVec[c];
    for(int zi = 0; zi < Nz; ++zi) {
      for(int yi = 0; yi < Ny; ++yi) {
        int bnd = (((zi*Ny) + yi)*Nx) + xi;
        int db = 0;
        int bid = (bnd*dofsPerNode) + db;
        arr[bid] = 0.0;
      }//end yi
    }//end zi
  }//end c 

  //y
  for(int c = 0; c < yVec.size(); ++c) {
    int yi = yVec[c];
    for(int zi = 0; zi < Nz; ++zi) {
      for(int xi = 0; xi < Nx; ++xi) {        
        int bnd = (((zi*Ny) + yi)*Nx) + xi;
        int db = 0;
        int bid = (bnd*dofsPerNode) + db;
        arr[bid] = 0.0;
      }//end xi
    }//end zi
  }//end c

  //z
  for(int c = 0; c < zVec.size(); ++c) {
    int zi = zVec[c];
    for(int yi = 0; yi < Ny; ++yi) {
      for(int xi = 0; xi < Nx; ++xi) {
        int bnd = (((zi*Ny) + yi)*Nx) + xi;
        int db = 0;
        int bid = (bnd*dofsPerNode) + db;
        arr[bid] = 0.0;
      }//end xi
    }//end yi
  }//end c
}

void printMatrix(MyMatrix & myMat) {
  for(size_t i = 0; i < ((myMat.nzCols).size()); ++i) {
    for(size_t j = 0; j < (((myMat.nzCols)[i]).size()); ++j) {
      std::cout<<"A["<<i<<"]["<<((myMat.nzCols)[i][j])<<"] = "<<std::setprecision(15)<<((myMat.vals)[i][j])<<std::endl;
    }//end j
    std::cout<<std::endl;
  }//end i
}

void assembleMatrix(MyMatrix & myMat, std::vector<std::vector<double> > const & elemMat, const unsigned int K, 
    const unsigned int dim, const unsigned int Nz, const unsigned int Ny, const unsigned int Nx) {
  unsigned int ye, ze;
  if(dim > 1) {
    assert(Ny > 1);
    ye = Ny - 1;
  } else {
    assert(Ny == 1);
    ye = 1;
  }
  if(dim > 2) {
    assert(Nz > 1);
    ze = Nz - 1;
  } else {
    assert(Nz == 1);
    ze = 1;
  }
  assert(dim > 0);
  assert(Nx > 1);

  size_t dofsPerNode = getDofsPerNode(dim, K); 
  std::cout<<"DofsPerNode = "<<dofsPerNode<<std::endl;

  unsigned int xe = Nx - 1;

  size_t matSz = dofsPerNode*Nz*Ny*Nx;
  std::cout<<"GlobalMatSize = "<<matSz<<std::endl;
  myMat.nzCols.clear();
  myMat.vals.clear();
  myMat.nzCols.resize(matSz);
  myMat.vals.resize(matSz);

  unsigned int nodesPerElem = (1 << dim);
  for(unsigned int zi = 0; zi < ze; ++zi) {
    for(unsigned int yi = 0; yi < ye; ++yi) {
      for(unsigned int xi = 0; xi < xe; ++xi) {
        for(unsigned int nr = 0, r = 0; nr < nodesPerElem; ++nr) {
          unsigned int zr = (nr/4);
          unsigned int yr = ((nr/2)%2);
          unsigned int xr = (nr%2);
          for(unsigned int dr = 0; dr < dofsPerNode; ++r, ++dr) {
            unsigned int row = ((((((zi + zr)*Ny) + (yi + yr))*Nx) + (xi + xr))*dofsPerNode) + dr;
            for(unsigned int nc = 0, c = 0; nc < nodesPerElem; ++nc) {
              unsigned int zc = (nc/4);
              unsigned int yc = ((nc/2)%2);
              unsigned int xc = (nc%2);
              for(unsigned int dc = 0; dc < dofsPerNode; ++c, ++dc) {
                unsigned int col = ((((((zi + zc)*Ny) + (yi + yc))*Nx) + (xi + xc))*dofsPerNode) + dc;
                std::vector<unsigned int>::iterator pos = std::lower_bound(((myMat.nzCols)[row]).begin(),
                    ((myMat.nzCols)[row]).end(), col);
                if(pos == (((myMat.nzCols)[row]).end())) {
                  ((myMat.vals)[row]).insert((((myMat.vals)[row]).end()), (elemMat[r][c]));
                  ((myMat.nzCols)[row]).insert(pos, col);
                } else if((*pos) == col) {
                  (myMat.vals)[row][(pos - (((myMat.nzCols)[row]).begin()))] += (elemMat[r][c]);
                } else {
                  ((myMat.vals)[row]).insert(((((myMat.vals)[row]).begin()) + (pos - (((myMat.nzCols)[row]).begin()))), (elemMat[r][c]));
                  ((myMat.nzCols)[row]).insert(pos, col);
                }
              }//end dc
            }//end nc
          }//end dr
        }//end nr
      }//end xi
    }//end yi
  }//end zi
}

void dirichletMatrixCorrection(MyMatrix & myMat, const unsigned int K, const unsigned int dim,
    const int Nz, const int Ny, const int Nx) {
  if(dim > 1) {
    assert(Ny > 1);
  } else {
    assert(Ny == 1);
  }
  if(dim > 2) {
    assert(Nz > 1);
  } else {
    assert(Nz == 1);
  }
  assert(dim > 0);
  assert(Nx > 1);

  int dofsPerNode = getDofsPerNode(dim, K); 

  std::vector<int> xVec; 
  xVec.push_back(0);
  xVec.push_back((Nx - 1));

  std::vector<int> yVec; 
  if(dim > 1) {
    yVec.push_back(0);
    yVec.push_back((Ny - 1));
  }

  std::vector<int> zVec;
  if(dim > 2) {
    zVec.push_back(0);
    zVec.push_back((Nz - 1));
  }

  //x
  for(int c = 0; c < xVec.size(); ++c) {
    int xi = xVec[c];
    for(int zi = 0; zi < Nz; ++zi) {
      for(int yi = 0; yi < Ny; ++yi) {
        std::vector<int> nh;
        getNeighbors(nh, zi, yi, xi, Nz, Ny, Nx);
        int bnd = (((zi*Ny) + yi)*Nx) + xi;
        int db = 0;
        int bid = (bnd*dofsPerNode) + db;
        for(int n = 0; n < nh.size(); ++n) {
          for(int dn = 0; dn < dofsPerNode; ++dn){
            int nid = (nh[n]*dofsPerNode) + dn;
            setValue(myMat, bid, nid, 0.0);
            setValue(myMat, nid, bid, 0.0);
          }//end dn
        }//end n
        for(int od = 1; od < dofsPerNode; ++od) {
          int oid = (bnd*dofsPerNode) + od;
          setValue(myMat, bid, oid, 0.0);
          setValue(myMat, oid, bid, 0.0);
        }//end od
        setValue(myMat, bid, bid, 1.0);
      }//end yi
    }//end zi
  }//end c 

  //y
  for(int c = 0; c < yVec.size(); ++c) {
    int yi = yVec[c];
    for(int zi = 0; zi < Nz; ++zi) {
      for(int xi = 0; xi < Nx; ++xi) {        
        std::vector<int> nh;
        getNeighbors(nh, zi, yi, xi, Nz, Ny, Nx);
        int bnd = (((zi*Ny) + yi)*Nx) + xi;
        int db = 0;
        int bid = (bnd*dofsPerNode) + db;
        for(int n = 0; n < nh.size(); ++n) {
          for(int dn = 0; dn < dofsPerNode; ++dn){
            int nid = (nh[n]*dofsPerNode) + dn;
            setValue(myMat, bid, nid, 0.0);
            setValue(myMat, nid, bid, 0.0);
          }//end dn
        }//end n
        for(int od = 1; od < dofsPerNode; ++od) {
          int oid = (bnd*dofsPerNode) + od;
          setValue(myMat, bid, oid, 0.0);
          setValue(myMat, oid, bid, 0.0);
        }//end od
        setValue(myMat, bid, bid, 1.0);
      }//end xi
    }//end zi
  }//end c

  //z
  for(int c = 0; c < zVec.size(); ++c) {
    int zi = zVec[c];
    for(int yi = 0; yi < Ny; ++yi) {
      for(int xi = 0; xi < Nx; ++xi) {
        std::vector<int> nh;
        getNeighbors(nh, zi, yi, xi, Nz, Ny, Nx);
        int bnd = (((zi*Ny) + yi)*Nx) + xi;
        int db = 0;
        int bid = (bnd*dofsPerNode) + db;
        for(int n = 0; n < nh.size(); ++n) {
          for(int dn = 0; dn < dofsPerNode; ++dn){
            int nid = (nh[n]*dofsPerNode) + dn;
            setValue(myMat, bid, nid, 0.0);
            setValue(myMat, nid, bid, 0.0);
          }//end dn
        }//end n
        for(int od = 1; od < dofsPerNode; ++od) {
          int oid = (bnd*dofsPerNode) + od;
          setValue(myMat, bid, oid, 0.0);
          setValue(myMat, oid, bid, 0.0);
        }//end od
        setValue(myMat, bid, bid, 1.0);
      }//end xi
    }//end yi
  }//end c
}

void createKrylovObject(ML_Krylov*& krylov_obj, ML* ml_obj, const unsigned int maxIters, const double rTol) {
  krylov_obj = ML_Krylov_Create(ml_obj->comm);
  //ML_GMRES does not work!
  ML_Krylov_Set_Method(krylov_obj, ML_CG);
  ML_Krylov_Set_Amatrix(krylov_obj, &((ml_obj->Amat)[0]));
  ML_Krylov_Set_PreconFunc(krylov_obj, ML_MGVSolve_Wrapper);
  ML_Krylov_Set_Precon(krylov_obj, ml_obj);
  ML_Krylov_Set_MaxIterations(krylov_obj, maxIters);
  ML_Krylov_Set_Tolerance(krylov_obj, rTol);
  ML_Krylov_Set_PrintFreq(krylov_obj, 1);
}

void createMLobjects(ML*& ml_obj, ML_Aggregate*& agg_obj, const unsigned int numGrids, const unsigned int maxIters, 
    const double rTol, const unsigned int dim, const unsigned int K, MyMatrix& myMat) {
  ML_set_random_seed(123456);

  ML_Create(&ml_obj, numGrids);
  ML_Init_Amatrix(ml_obj, 0, ((myMat.vals).size()), ((myMat.vals).size()), &myMat);
  ML_Set_Amatrix_Getrow(ml_obj, 0, &myGetRow, NULL, ((myMat.vals).size()));
  ML_Set_Amatrix_Matvec(ml_obj, 0, &myMatVec);
  ML_Set_MaxIterations(ml_obj, maxIters);
  ML_Set_Tolerance(ml_obj, rTol);
  ML_Set_ResidualOutputFrequency(ml_obj, 1);
  ML_Set_PrintLevel(10);
  ML_Set_OutputLevel(ml_obj, 10);

  unsigned int numPDEs = (K + 1); //DOFs per node;
  if(dim > 1) {
    numPDEs *= (K + 1);
  } 
  if(dim > 2) {
    numPDEs *= (K + 1);
  }
  unsigned int coarseSize = 3*numPDEs; //2 Elements per dim;
  if(dim > 1) {
    coarseSize *= 3;
  } 
  if(dim > 2) {
    coarseSize *= 3;
  }

  ML_Aggregate_Create(&agg_obj);
  agg_obj->num_PDE_eqns = numPDEs;
  agg_obj->nullspace_dim = 1; 
  ML_Aggregate_Set_MaxCoarseSize(agg_obj, coarseSize);

  const unsigned int nlevels = ML_Gen_MGHierarchy_UsingAggregation(ml_obj, 0, ML_INCREASING, agg_obj);
  std::cout<<"Number of actual MG levels: "<<nlevels<<std::endl;

  for(int lev = 0; lev < (nlevels - 1); ++lev) {
    ML_Gen_Smoother_SymGaussSeidel(ml_obj, lev, ML_BOTH, 2, 1.0);
  }
  ML_Gen_Smoother_Amesos(ml_obj, (nlevels - 1), ML_AMESOS_KLU, -1, 0.0);

  ML_Gen_Solver(ml_obj, ML_MGV, 0, (nlevels-1));
}

void destroyMLobjects(ML*& ml_obj, ML_Aggregate*& agg_obj) {
  ML_Aggregate_Destroy(&agg_obj);
  ML_Destroy(&ml_obj);
}

void computeRandomRHS(double* rhsArr, MyMatrix & myMat, const unsigned int K, const unsigned int dim,
    const int Nz, const int Ny, const int Nx) {
  const unsigned int len = (myMat.vals).size();
  double* tmpSol = new double[len];
  for(unsigned int i = 0; i < len; ++i) {
    tmpSol[i] = (static_cast<double>(rand()))/(static_cast<double>(RAND_MAX));
  }//end for i
  zeroBoundaries(tmpSol, K, dim, Nz, Ny, Nx);
  myMatVecPrivate(&myMat, len, tmpSol, rhsArr);
  delete [] tmpSol;
}

int myGetRow(ML_Operator* data, int N_requested_rows, int requested_rows[],
    int allocated_space, int columns[], double values[], int row_lengths[]) {
  MyMatrix* myMat = reinterpret_cast<MyMatrix*>(ML_Get_MyGetrowData(data));
  int spaceRequired = 0;
  int cnt = 0;
  for(int i = 0; i < N_requested_rows; ++i) {
    int row = requested_rows[i];
    spaceRequired += ((myMat->nzCols)[row]).size();
    if(allocated_space >= spaceRequired) {
      for(size_t j = 0; j < ((myMat->nzCols)[row]).size(); ++j) {
        columns[cnt] = (myMat->nzCols)[row][j];
        values[cnt] = (myMat->vals)[row][j];
        ++cnt;
      }//end for j
      row_lengths[i] = ((myMat->nzCols)[row]).size();
    } else {
      return 0;
    }
  }//end for i
  return 1;
}

int myMatVec(ML_Operator* data, int in_length, double in[], int out_length, double out[]) {
  MyMatrix* myMat = reinterpret_cast<MyMatrix*>(ML_Get_MyMatvecData(data));
  myMatVecPrivate(myMat, out_length, in, out); 
  return 0;
}

void myMatVecPrivate(MyMatrix* myMat, const unsigned int len, double* in, double* out) {
  for(int i = 0; i < len; ++i) {
    out[i] = 0.0;
    for(size_t j = 0; j < ((myMat->nzCols)[i]).size(); ++j) {
      out[i] += ( ((myMat->vals)[i][j]) * (in[(myMat->nzCols)[i][j]]) );
    }//end for j
  }//end for i
}

void getNeighbors(std::vector<int> & nh, int zi, int yi, int xi, int Nz, int Ny, int Nx) {
  nh.clear();
  for(int k = -1; k < 2; ++k) {
    int zo = zi + k;
    if( (zo >= 0) && (zo < Nz) ) {
      for(int j = -1; j < 2; ++j) {
        int yo = yi + j;
        if( (yo >= 0) && (yo < Ny) ) {
          for(int i = -1; i < 2; ++i) {
            int xo = xi + i;
            if( (xo >= 0) && (xo < Nx) ) {
              if( k || j || i ) {
                int oth = (((zo*Ny) + yo)*Nx) + xo;
                nh.push_back(oth);
              }
            }
          }//end i
        }
      }//end j
    }
  }//end k
}

void setValue(MyMatrix & myMat, unsigned int row, unsigned int col, double val) {
  std::vector<unsigned int>::iterator pos = std::lower_bound(((myMat.nzCols)[row]).begin(),
      ((myMat.nzCols)[row]).end(), col);
  assert(pos != (((myMat.nzCols)[row]).end()));
  assert((*pos) == col);
  (myMat.vals)[row][(pos - (((myMat.nzCols)[row]).begin()))] = val;
}



