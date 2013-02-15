
#ifndef __ASSEMBLY__
#define __ASSEMBLY__

void buildKmat(std::vector<Mat>& Kmat, std::vector<DM>& da, bool print);

void assembleKmat(int dim, std::vector<PetscInt>& Nz, std::vector<PetscInt>& Ny, std::vector<PetscInt>& Nx,
    std::vector<Mat>& Kmat, std::vector<DM>& da, int K, std::vector<long long int>& coeffs,
    std::vector<unsigned long long int>& factorialsList, bool print);

void computeKmat(Mat Kmat, DM da, std::vector<std::vector<long double> >& elemMat);

#endif

