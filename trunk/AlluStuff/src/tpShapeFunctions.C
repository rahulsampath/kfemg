
#include "tpShapeFunctions.h"
#include <cstdio>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void tensorProduct3D(int K,  const std::vector<double> & si,const std::vector<double> &eta,
    const std::vector<double> &zeta, std::vector<double> &sietazeta , int dsi, int deta, int dzeta )
{
  //
  //     tensor product of 1d si[ ] , eta[ ]  , zeta[ ] functions to genareate 
  //     3d functiond in sietazeta[ ] : standard setup of dofs
  //

  int n1b_zeta, n1e_zeta, n1b_eta, n1e_eta, n1b_si, n1e_si;
  int n2b_zeta, n2e_zeta, n2b_eta, n2e_eta, n2b_si, n2e_si;
  ///////////////////////////////////////////////////////////////
  //    
  //       8            7
  //        o-----------o
  //       /:          /|
  //      / :         / |
  //   5 /  :      6 /  |
  //    o-----------o   |
  //    |   o.......|...o 3
  //    |  .4       |  /
  //    | .         | /
  //    |.          |/
  //    o-----------o
  //    1           2
  //
  ////////////////////////////////////////////////////////////////

  int ndf_si, ndf_eta, ndf_zeta;

  ndf_si   = K + 1;
  ndf_eta  = K + 1;
  ndf_zeta = K + 1;

  n1b_si = (2*K+2)*dsi; 
  n1e_si = n1b_si + ndf_si;
  n2b_si = n1e_si;
  n2e_si = n2b_si + ndf_si;

  n1b_eta = (2*K+2)*deta; 
  n1e_eta = n1b_eta + ndf_eta;
  n2b_eta = n1e_eta;
  n2e_eta = n2b_eta + ndf_eta;

  n1b_zeta = (2*K+2)*dzeta; 
  n1e_zeta = n1b_zeta + ndf_zeta;
  n2b_zeta = n1e_zeta;
  n2e_zeta = n2b_zeta + ndf_zeta;

  /*
  printf("n1b_si : %d n1e_si : %d\n", n1b_si, n1e_si); 
  printf("n2b_si : %d n2e_si : %d\n", n2b_si, n2e_si); 

  printf("n1b_eta : %d n1e_eta : %d\n", n1b_eta, n1e_eta); 
  printf("n2b_eta : %d n2e_eta : %d\n", n2b_eta, n2e_eta); 

  printf("n1b_zeta : %d n1e_zeta : %d\n", n1b_zeta, n1e_zeta); 
  printf("n2b_zeta : %d n2e_zeta : %d\n", n2b_zeta, n2e_zeta); 
  
  */

  printf("\n <<<<< entering tensorProduct3D >>>>>\n");
  //
  int kk = -1;
  //
  //-------- node 1 -----------
  //
  for(int k=n1b_zeta  ; k<n1e_zeta ; ++k){
    for(int j=n1b_eta ; j<n1e_eta  ; ++j){
      for(int i=n1b_si; i<n1e_si   ; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  //
  //-------- node 2 -----------
  //
  for(int  k=n1b_zeta  ; k< n1e_zeta; ++k){
    for(int  j=n1b_eta ; j< n1e_eta; ++j){
      for(int  i=n2b_si; i< n2e_si; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  //
  //-------- node 3 -----------
  //
  for(int  k=n1b_zeta  ; k<n1e_zeta; ++k){
    for(int  j=n2b_eta ; j<n2e_eta  ; ++j){
      for(int  i=n2b_si; i<n2e_si    ; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  //
  //-------- node 4 -----------
  //
  for(int  k=n1b_zeta  ; k<n1e_zeta; ++k){
    for(int  j=n2b_eta ; j<n2e_eta  ; ++j){
      for(int  i=n1b_si; i<n1e_si    ; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  //
  //-------- node 5 -----------
  //
  for(int  k=n2b_zeta  ; k<n2e_zeta; ++k){
    for(int  j=n1b_eta ; j<n1e_eta  ; ++j){
      for(int  i=n1b_si; i<n1e_si    ; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  //
  //-------- node 6 -----------
  //
  for(int  k=n2b_zeta  ; k<n2e_zeta; ++k){
    for(int  j=n1b_eta ; j<n1e_eta  ; ++j){
      for(int  i=n2b_si; i<n2e_si    ; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  //
  //-------- node 7 -----------
  //
  for(int  k=n2b_zeta  ; k<n2e_zeta; ++k){
    for(int  j=n2b_eta ; j<n2e_eta  ; ++j){
      for(int  i=n2b_si; i<n2e_si    ; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  //
  //-------- node 8 -----------
  //
  for(int  k=n2b_zeta  ; k<n2e_zeta; ++k){
    for(int  j=n2b_eta ; j<n2e_eta  ; ++j){
      for(int  i=n1b_si; i<n1e_si    ; ++i){
        kk = kk + 1;
        sietazeta[ kk ] = si[ i ] * eta[ j ] * zeta[ k ];
      }
    }
  }
  
  printf("\n <<<<< leaving tensorProduct3D >>>>>\n " );
  
  return;
}
