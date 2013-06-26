
#include "gmg/include/static/mgPC.h"
#include "gmg/include/boundary.h"
#include "gmg/include/gmgUtils.h"
#include "gmg/include/intergrid.h"

#ifdef DEBUG
#include <cassert>
#endif

void setupMG(PC pc, int K, int currLev, std::vector<std::vector<DM> >& da,
    std::vector<std::vector<long long int> >& coeffs, std::vector<std::vector<Mat> >& Kmat,
    std::vector<std::vector<Mat> >& Pmat, std::vector<std::vector<Vec> >& tmpCvec) {
#ifdef DEBUG
  assert(currLev > 0);
#endif
}



