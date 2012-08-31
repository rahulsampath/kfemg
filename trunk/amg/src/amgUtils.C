
#include <cassert>
#include "ml_include.h"
#include "amg/include/amgUtils.h"

int myMatVec(ML_Operator *data, int in_length, double in[], int out_length, double out[]) {
  MyMatrix* myMat = reinterpret_cast<MyMatrix*>(ML_Get_MyMatvecData(data));
  for(int i = 0; i < out_length; ++i) {
    out[i] = 0.0;
    for(size_t j = 0; j < ((myMat->nzCols)[i]).size(); ++j) {
      out[i] += ( ((myMat->vals)[i][j]) * (in[(myMat->nzCols)[i][j]]) );
    }//end for j
  }//end for i
  return 0;
}

int myGetRow(ML_Operator *data, int N_requested_rows, int requested_rows[],
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


