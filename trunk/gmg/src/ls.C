
#include "gmg/include/ls.h"

void setupLS(LSdata* data, Mat Kmat) {
  data->Kmat = Kmat;
}

void destroyLS(LSdata* data) {
  delete data;
}

