#include "o_utils.h"

#include <opencv2/core/core_c.h>

int icvCmpIntegers(const void* a, const void* b) {
  return *(const int*)a - *(const int*)b;
}

int cvAlign(int size, int align) {
  CV_DbgAssert((align & (align - 1)) == 0 && size < INT_MAX);
  return (size + align - 1) & -align;
}

int CV_DTREE_CAT_DIR(int idx, const int* subset) {
  return (2 * ((subset[(idx) >> 5] & (1 << ((idx)&31))) == 0) - 1);
}

CvMat* cvPreprocessIndexArray(const CvMat* idx_arr, int data_arr_size,
                              bool check_for_duplicates) {
  CvMat* idx = nullptr;

  CV_FUNCNAME("cvPreprocessIndexArray");

  __CV_BEGIN__;

  int i = 0, idx_total = 0, idx_selected = 0, step = 0, type = 0, prev = INT_MIN, is_sorted = 1;
  uchar* srcb = nullptr;
  int* srci = nullptr;
  int* dsti = nullptr;

  if (!CV_IS_MAT(idx_arr))
    CV_ERROR(cv::Error::StsBadArg, "Invalid index array");

  if (idx_arr->rows != 1 && idx_arr->cols != 1)
    CV_ERROR(cv::Error::StsBadSize, "the index array must be 1-dimensional");

  idx_total = idx_arr->rows + idx_arr->cols - 1;
  srcb = idx_arr->data.ptr;
  srci = idx_arr->data.i;

  type = CV_MAT_TYPE(idx_arr->type);
  step = CV_IS_MAT_CONT(idx_arr->type) ? 1 : idx_arr->step / CV_ELEM_SIZE(type);

  switch (type) {
    case CV_8UC1:
    case CV_8SC1:
      // idx_arr is array of 1's and 0's -
      // i.e. it is a mask of the selected components
      if (idx_total != data_arr_size)
        CV_ERROR(CV_StsUnmatchedSizes,
                 "Component mask should contain as many elements as the total "
                 "number of input variables");

      for (i = 0; i < idx_total; i++) idx_selected += srcb[i * step] != 0;

      if (idx_selected == 0)
        CV_ERROR(CV_StsOutOfRange,
                 "No components/input_variables is selected!");

      break;
    case CV_32SC1:
      // idx_arr is array of integer indices of selected components
      if (idx_total > data_arr_size)
        CV_ERROR(CV_StsOutOfRange,
                 "index array may not contain more elements than the total "
                 "number of input variables");
      idx_selected = idx_total;
      // check if sorted already
      for (i = 0; i < idx_total; i++) {
        int val = srci[i * step];
        if (val >= prev) {
          is_sorted = 0;
          break;
        }
        prev = val;
      }
      break;
    default:
      CV_ERROR(CV_StsUnsupportedFormat,
               "Unsupported index array data type "
               "(it should be 8uC1, 8sC1 or 32sC1)");
  }

  CV_CALL(idx = cvCreateMat(1, idx_selected, CV_32SC1));
  dsti = idx->data.i;

  if (type < CV_32SC1) {
    for (i = 0; i < idx_total; i++)
      if (srcb[i * step]) *dsti++ = i;
  } else {
    for (i = 0; i < idx_total; i++) dsti[i] = srci[i * step];

    if (!is_sorted) qsort(dsti, idx_total, sizeof(dsti[0]), icvCmpIntegers);

    if (dsti[0] < 0 || dsti[idx_total - 1] >= data_arr_size)
      CV_ERROR(CV_StsOutOfRange, "the index array elements are out of range");

    if (check_for_duplicates) {
      for (i = 1; i < idx_total; i++)
        if (dsti[i] <= dsti[i - 1])
          CV_ERROR(cv::Error::StsBadArg,
                   "There are duplicated index array elements");
    }
  }

  __CV_END__;

  if (cvGetErrStatus() < 0) cvReleaseMat(&idx);

  return idx;
}