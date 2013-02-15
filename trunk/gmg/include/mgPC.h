
#ifndef __MG_PC__
#define __MG_PC__

struct MGdata {
  PetscInt numVcycles;
  std::vector<Mat> Kmat;
  std::vector<Mat> Pmat;
  std::vector<Vec> tmpCvec; 
  std::vector<KSP> smoother;
  KSP coarseSolver;
  std::vector<Vec> mgSol;
  std::vector<Vec> mgRhs;
  std::vector<Vec> mgRes;
};

PetscErrorCode applyMG(PC pc, Vec in, Vec out);

void applyVcycle(int currLev, std::vector<Mat>& Kmat, std::vector<Mat>& Pmat, 
    std::vector<Vec>& tmpCvec, std::vector<KSP>& smoother, KSP coarseSolver,
    std::vector<Vec>& mgSol, std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

void buildMGworkVecs(std::vector<Mat>& Kmat, std::vector<Vec>& mgSol, 
    std::vector<Vec>& mgRhs, std::vector<Vec>& mgRes);

#endif


