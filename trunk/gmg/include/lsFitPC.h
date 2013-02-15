
#ifndef __LS_FIT_PC__
#define __LS_FIT_PC__

#include <vector>

void computeFxPhi1D(int mode, int Nx, int K, std::vector<long long int>& coeffs,
    std::vector<double>& res);

void computeLSfit(double aVec[2], double HmatInv[2][2], std::vector<double>& fVec,
    std::vector<double>& gVec, std::vector<double>& cVec);

double computeRval(double aVec[2], std::vector<double>& fVec, std::vector<double>& gVec, 
    std::vector<double>& cVec);

void computeJvec(double jVec[2], double aVec[2], std::vector<double>& fVec,
    std::vector<double>& gVec, std::vector<double>& cVec);

void computeHmat(double mat[2][2], std::vector<double>& gVec, std::vector<double>& cVec);

#endif


