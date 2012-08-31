
#ifndef __AMG_UTILS__
#define __AMG_UTILS__

#include <vector>
#include "ml_include.h"

struct MyMatrix {
  std::vector<std::vector<unsigned int> > nzCols;
  std::vector<std::vector<double> > vals;
};

int myMatVec(ML_Operator *data, int in_length, double in[], int out_length, double out[]);

int myGetRow(ML_Operator *data, int N_requested_rows, int requested_rows[],
    int allocated_space, int columns[], double values[], int row_lengths[]);

#endif


