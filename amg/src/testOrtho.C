
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include "mpi.h"
#include "ml_include.h"
#include "common/include/commonUtils.h"
#include "amg/include/amgUtils.h"

void setInputVector(const unsigned int waveNum, const unsigned int waveDof,
    const unsigned int K, const unsigned int Nx, double* inArr) {
  for(int i = 0; i < ((K + 1)*Nx); ++i) {
    inArr[i] = 0.0;
  }//end i

  if(waveNum == 0) {
    inArr[waveDof] = 1.0;
  } else if(waveNum == (Nx - 1)) {
    inArr[((K + 1)*(Nx - 1)) + waveDof] = 1.0;
  } else {
    for(int i = 0; i < Nx; ++i) {
      double fac = (static_cast<double>(i*waveNum))/(static_cast<double>(Nx - 1));
      inArr[((K + 1)*i) + waveDof] = sin(fac*__PI__);
    }//end i
    suppressSmallValues(((K + 1)*Nx), inArr);
  }
}

/*
   void setInputVector(const unsigned int waveNum, const unsigned int waveDof,
   const unsigned int K, const unsigned int Nx, double* inArr) {
   for(int i = 0; i < ((K + 1)*Nx); ++i) {
   inArr[i] = 0.0;
   }//end i

   if((waveDof == 0) && (waveNum == 0)) {
   inArr[0] = 1.0;
   } else if((waveDof == 0) && (waveNum == (Nx - 1))) {
   inArr[((K + 1)*(Nx - 1))] = 1.0;
   } else {
   for(int i = 0; i < Nx; ++i) {
   double fac = (static_cast<double>(i*waveNum))/(static_cast<double>(Nx - 1));
   if(waveDof == 0) {
   inArr[((K + 1)*i)] = sin(fac*__PI__);
   } else {
   inArr[((K + 1)*i) + waveDof] = cos(fac*__PI__);
   }
   }//end i
   suppressSmallValues(((K + 1)*Nx), inArr);
   }
   }
   */

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  if(argc <= 2) {
    std::cout<<"USAGE: <exe> K Nxf "<<std::endl;
    assert(false);
  }
  const unsigned int dim = 1; 
  std::cout<<"Dim = "<<dim<<std::endl;

  const unsigned int K = atoi(argv[1]);
  std::cout<<"K = "<<K<<std::endl;
  assert(K <= 7);

  const unsigned int Nxf = atoi(argv[2]); 
  assert(Nxf > 1);
  std::cout<<"Nxf = "<<Nxf<<std::endl;
  const unsigned int Nxc = 1 + ((Nxf - 1)/2);
  std::cout<<"Nxc = "<<Nxc<<std::endl;

  const unsigned int dofsPerNode = K + 1;
  unsigned int numGaussPts = (2*K) + 2;

  std::vector<long double> gPt(numGaussPts);
  std::vector<long double> gWt(numGaussPts);
  gaussQuad(gPt, gWt);

  long double hxf = 1.0L/(static_cast<long double>(Nxf - 1));
  long double hxc = 2.0*hxf;
  std::cout<<"hxf = "<<(std::setprecision(13))<<hxf<<std::endl;

  std::vector<long long int> coeffs;
  read1DshapeFnCoeffs(K, coeffs);

  std::vector<unsigned long long int> factorialsList;
  initFactorials(factorialsList);

  double* vec = new double[Nxf*dofsPerNode];

  for(int wNum = 0; wNum < Nxf; ++wNum) {
    std::cout<<"wNum = "<<wNum<<std::endl;
    for(int wDof = 0; wDof <= K; ++wDof) {
      std::cout<<"wDof = "<<wDof<<std::endl;

      if(wDof == 0) {
        if((wNum == 0) || (wNum == (Nxf - 1))) {
          continue;
        }
      }

      setInputVector(wNum, wDof, K, Nxf, vec);

      double maxRes = 0.0;
      for(int j = 0; j < Nxc; ++j) {
        for(int dj = 0; dj <= K; ++dj) {
          if(dj == 0) {
            if((j == 0) || (j == (Nxc - 1))) {
              continue;
            }
          }
          double res = 0.0;
          if(j > 0) {
            int cNd = 1;
            //Elem 0
            {
              int fId = (2*j) - 2;
              for(int g = 0; g < numGaussPts; ++g) {
                double glob = coordLocalToGlobal(gPt[g], (fId*hxf), hxf);
                double loc = coordGlobalToLocal(glob, ((j - cNd)*hxc), hxc);
                double v = 0;
                for(int fNd = 0; fNd < 2; ++fNd) {
                  for(int df = 0; df <= K; ++df) {
                    v += ( vec[((fId + fNd)*dofsPerNode) + df] * eval1DshFn(fNd, df, K, coeffs, gPt[g]) );
                  }//end df
                }//end fNd
                res += ( gWt[g] * v * eval1DshFn(cNd, dj, K, coeffs, loc) );
              }//end g
            }
            //Elem 1
            {
              int fId = (2*j) - 1;
              for(int g = 0; g < numGaussPts; ++g) {
                double glob = coordLocalToGlobal(gPt[g], (fId*hxf), hxf);
                double loc = coordGlobalToLocal(glob, ((j - cNd)*hxc), hxc);
                double v = 0;
                for(int fNd = 0; fNd < 2; ++fNd) {
                  for(int df = 0; df <= K; ++df) {
                    v += ( vec[((fId + fNd)*dofsPerNode) + df] * eval1DshFn(fNd, df, K, coeffs, gPt[g]) );
                  }//end df
                }//end fNd
                res += ( gWt[g] * v * eval1DshFn(cNd, dj, K, coeffs, loc) );
              }//end g
            }
          }
          if(j < (Nxc - 1)) {
            int cNd = 0;
            //Elem 2
            {
              int fId = (2*j);
              for(int g = 0; g < numGaussPts; ++g) {
                double glob = coordLocalToGlobal(gPt[g], (fId*hxf), hxf);
                double loc = coordGlobalToLocal(glob, ((j - cNd)*hxc), hxc);
                double v = 0;
                for(int fNd = 0; fNd < 2; ++fNd) {
                  for(int df = 0; df <= K; ++df) {
                    v += ( vec[((fId + fNd)*dofsPerNode) + df] * eval1DshFn(fNd, df, K, coeffs, gPt[g]) );
                  }//end df
                }//end fNd
                res += ( gWt[g] * v * eval1DshFn(cNd, dj, K, coeffs, loc) );
              }//end g
            }
            //Elem 3
            {
              int fId = (2*j) + 1;
              for(int g = 0; g < numGaussPts; ++g) {
                double glob = coordLocalToGlobal(gPt[g], (fId*hxf), hxf);
                double loc = coordGlobalToLocal(glob, ((j - cNd)*hxc), hxc);
                double v = 0;
                for(int fNd = 0; fNd < 2; ++fNd) {
                  for(int df = 0; df <= K; ++df) {
                    v += ( vec[((fId + fNd)*dofsPerNode) + df] * eval1DshFn(fNd, df, K, coeffs, gPt[g]) );
                  }//end df
                }//end fNd
                res += ( gWt[g] * v * eval1DshFn(cNd, dj, K, coeffs, loc) );
              }//end g
            }
          }
          //      std::cout<<"("<<j<<", "<<dj<<") = "<<(std::setprecision(13))<<res<<std::endl;
          if(maxRes < fabs(res)) {
            maxRes = fabs(res);
          }
        }//end dj
      }//end j

      std::cout<<"maxRes = "<<(std::setprecision(13))<<maxRes<<std::endl;
    }//end wDof
  }//end wNum

  delete [] vec;

  MPI_Finalize();

  return 0;
}  


