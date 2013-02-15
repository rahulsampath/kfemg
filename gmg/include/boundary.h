
#ifndef __BOUNDARY__
#define __BOUNDARY__

void correctKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, int K);

void dirichletMatrixCorrection(Mat Kmat, DM da, const int K);

#endif

