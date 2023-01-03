#include "o_cvdtreetraindata.h"

#include <cstdio>

#include <opencv2/core/core_c.h>

#include <cmath>

#include <opencv2/ml/ml.hpp>

#include "o_cvdtreenode.h"
#include "o_cvdtreesplit.h"
#include "o_utils.h"

static const float ord_nan = FLT_MAX * 0.5f;
static const int min_block_size = 1 << 16;
static const int block_size_delta = 1 << 10;

void cvCheckTrainData(const CvMat* train_data, int tflag,
                      const CvMat* missing_mask, int* var_all,
                      int* sample_all) {
  CV_FUNCNAME("cvCheckTrainData");

  if (var_all) *var_all = 0;

  if (sample_all) *sample_all = 0;

  __CV_BEGIN__;

  // check parameter types and sizes
  if (!CV_IS_MAT(train_data) || CV_MAT_TYPE(train_data->type) != CV_32FC1)
    CV_ERROR(cv::Error::StsBadArg, "train data must be floating-point matrix");

  if (missing_mask) {
    if (!CV_IS_MAT(missing_mask) || !CV_IS_MASK_ARR(missing_mask) ||
        !CV_ARE_SIZES_EQ(train_data, missing_mask))
      CV_ERROR(cv::Error::StsBadArg,
               "missing value mask must be 8-bit matrix of the same size as "
               "training data");
  }

  if (tflag != cv::ml::ROW_SAMPLE && tflag != cv::ml::COL_SAMPLE)
    CV_ERROR(cv::Error::StsBadArg,
             "Unknown training data layout (must be cv::ml::ROW_SAMPLE or "
             "cv::ml::COL_SAMPLE)");

  if (var_all)
    *var_all =
        tflag == cv::ml::ROW_SAMPLE ? train_data->cols : train_data->rows;

  if (sample_all)
    *sample_all =
        tflag == cv::ml::ROW_SAMPLE ? train_data->rows : train_data->cols;

  __CV_END__;
}

CvDTreeTrainData::CvDTreeTrainData() {
  var_idx = var_type = cat_count = cat_ofs = cat_map = priors = priors_mult =
      counts = direction = split_buf = responses_copy = nullptr;
  buf = nullptr;
  tree_storage = temp_storage = nullptr;
  work_var_count = 0;
  tflag = cv::ml::ROW_SAMPLE;
  train_data = nullptr;
  responses = nullptr;
  is_buf_16u = false;

  clear();
}

CvDTreeTrainData::CvDTreeTrainData(
    const CvMat* _train_data, int _tflag, const CvMat* _responses,
    const CvMat* _var_idx, const CvMat* _sample_idx, const CvMat* _var_type,
    const CvMat* _missing_mask, const CvDTreeParams& _params, bool _shared,
    bool _add_labels) {
  var_idx = var_type = cat_count = cat_ofs = cat_map = priors = priors_mult =
      counts = direction = split_buf = responses_copy = nullptr;
  buf = nullptr;

  tree_storage = temp_storage = nullptr;

  set_data(_train_data, _tflag, _responses, _var_idx, _sample_idx, _var_type,
           _missing_mask, _params, _shared, _add_labels);
}

CvDTreeTrainData::~CvDTreeTrainData() { clear(); }

bool CvDTreeTrainData::set_params(const CvDTreeParams& _params) {
  bool ok = false;

  CV_FUNCNAME("CvDTreeTrainData::set_params");

  __CV_BEGIN__;

  // set parameters
  params = _params;

  if (params.max_categories < 2)
    CV_ERROR(CV_StsOutOfRange, "params.max_categories should be >= 2");
  params.max_categories = MIN(params.max_categories, 15);

  if (params.max_depth < 0)
    CV_ERROR(CV_StsOutOfRange, "params.max_depth should be >= 0");
  params.max_depth = MIN(params.max_depth, 25);

  params.min_sample_count = MAX(params.min_sample_count, 1);

  if (params.cv_folds < 0)
    CV_ERROR(CV_StsOutOfRange,
             "params.cv_folds should be =0 (the tree is not pruned) "
             "or n>0 (tree is pruned using n-fold cross-validation)");

  if (params.cv_folds == 1) params.cv_folds = 0;

  if (params.regression_accuracy < 0)
    CV_ERROR(CV_StsOutOfRange, "params.regression_accuracy should be >= 0");

  ok = true;

  __CV_END__;

  return ok;
}

struct CvPair16u32s {
  unsigned short* u;
  int* i;
};

class LessThanPairs {
 public:
  bool operator()(const CvPair16u32s& a, const CvPair16u32s& b) const {
    return *a.i < *b.i;
  }
};

CvMat* cvPreprocessVarType(const CvMat* var_type, const CvMat* var_idx,
                           int var_count, int* response_type) {
  CvMat* out_var_type = nullptr;
  CV_FUNCNAME("cvPreprocessVarType");

  if (response_type) *response_type = -1;

  __CV_BEGIN__;

  int i = 0, tm_size = 0, tm_step = 0;
  // int* map = 0;
  const uchar* src = nullptr;
  uchar* dst = nullptr;

  if (!CV_IS_MAT(var_type))
    CV_ERROR(var_type ? cv::Error::StsBadArg : cv::Error::StsNullPtr,
             "Invalid or absent var_type array");

  if (var_type->rows != 1 && var_type->cols != 1)
    CV_ERROR(CV_StsBadSize, "var_type array must be 1-dimensional");

  if (!CV_IS_MASK_ARR(var_type))
    CV_ERROR(CV_StsUnsupportedFormat, "type mask must be 8uC1 or 8sC1 array");

  tm_size = var_type->rows + var_type->cols - 1;
  tm_step =
      var_type->rows == 1 ? 1 : var_type->step / CV_ELEM_SIZE(var_type->type);

  if (/*tm_size != var_count &&*/ tm_size != var_count + 1)
    CV_ERROR(cv::Error::StsBadArg,
             "type mask must be of <input var count> + 1 size");

  if (response_type && tm_size > var_count)
    *response_type = var_type->data.ptr[var_count * tm_step] != 0;

  if (var_idx) {
    if (!CV_IS_MAT(var_idx) || CV_MAT_TYPE(var_idx->type) != CV_32SC1 ||
        (var_idx->rows != 1 && var_idx->cols != 1) ||
        !CV_IS_MAT_CONT(var_idx->type))
      CV_ERROR(
          cv::Error::StsBadArg,
          "var index array should be continuous 1-dimensional integer vector");
    if (var_idx->rows + var_idx->cols - 1 > var_count)
      CV_ERROR(CV_StsBadSize, "var index array is too large");
    // map = var_idx->data.i;
    var_count = var_idx->rows + var_idx->cols - 1;
  }

  CV_CALL(out_var_type = cvCreateMat(1, var_count, CV_8UC1));
  src = var_type->data.ptr;
  dst = out_var_type->data.ptr;

  for (i = 0; i < var_count; i++) {
    // int idx = map ? map[i] : i;
    assert((unsigned)/*idx*/ i < (unsigned)tm_size);
    dst[i] = (uchar)(src[/*idx*/ i * tm_step] != 0);
  }

  __CV_END__;

  return out_var_type;
}

void CvDTreeTrainData::set_data(const CvMat* _train_data, int _tflag,
                                const CvMat* _responses, const CvMat* _var_idx,
                                const CvMat* _sample_idx,
                                const CvMat* _var_type,
                                const CvMat* _missing_mask,
                                const CvDTreeParams& _params, bool _shared,
                                bool _add_labels, bool _update_data) {
  CvMat* sample_indices = nullptr;
  CvMat* var_type0 = nullptr;
  CvMat* tmp_map = nullptr;
  int** int_ptr = nullptr;
  CvPair16u32s* pair16u32s_ptr = nullptr;
  CvDTreeTrainData* data = nullptr;
  float* _fdst = nullptr;
  int* _idst = nullptr;
  unsigned short* udst = nullptr;
  int* idst = nullptr;

  CV_FUNCNAME("CvDTreeTrainData::set_data");

  __CV_BEGIN__;

  int sample_all = 0, r_type = 0, cv_n = 0;
  int total_c_count = 0;
  int tree_block_size = 0, temp_block_size = 0, max_split_size = 0, nv_size = 0, cv_size = 0;
  int ds_step = 0, dv_step = 0, ms_step = 0,
                        mv_step = 0;  // {data|mask}{sample|var}_step
  int vi = 0, i = 0, size = 0;
  char err[100];
  const int *sidx = nullptr, *vidx = nullptr;

  uint64 effective_buf_size = 0;
  int effective_buf_height = 0, effective_buf_width = 0;

  if (_update_data && data_root) {
    data = new CvDTreeTrainData(_train_data, _tflag, _responses, _var_idx,
                                _sample_idx, _var_type, _missing_mask, _params,
                                _shared, _add_labels);

    // compare new and old train data
    if (!(data->var_count == var_count &&
          cvNorm(data->var_type, var_type, CV_C) < FLT_EPSILON &&
          cvNorm(data->cat_count, cat_count, CV_C) < FLT_EPSILON &&
          cvNorm(data->cat_map, cat_map, CV_C) < FLT_EPSILON))
      CV_ERROR(cv::Error::StsBadArg,
               "The new training data must have the same types and the input "
               "and output variables "
               "and the same categories for categorical variables");

    cvReleaseMat(&priors);
    cvReleaseMat(&priors_mult);
    cvReleaseMat(&buf);
    cvReleaseMat(&direction);
    cvReleaseMat(&split_buf);
    cvReleaseMemStorage(&temp_storage);

    priors = data->priors;
    data->priors = nullptr;
    priors_mult = data->priors_mult;
    data->priors_mult = nullptr;
    buf = data->buf;
    data->buf = nullptr;
    buf_count = data->buf_count;
    buf_size = data->buf_size;
    sample_count = data->sample_count;

    direction = data->direction;
    data->direction = nullptr;
    split_buf = data->split_buf;
    data->split_buf = nullptr;
    temp_storage = data->temp_storage;
    data->temp_storage = nullptr;
    nv_heap = data->nv_heap;
    cv_heap = data->cv_heap;

    data_root = new_node(nullptr, sample_count, 0, 0);
    __CV_EXIT__;
  }

  clear();

  var_all = 0;
  rng = &cv::theRNG();

  CV_CALL(set_params(_params));

  // check parameter types and sizes
  CV_CALL(cvCheckTrainData(_train_data, _tflag, _missing_mask, &var_all,
                           &sample_all));

  train_data = _train_data;
  responses = _responses;

  if (_tflag == cv::ml::ROW_SAMPLE) {
    ds_step = _train_data->step / CV_ELEM_SIZE(_train_data->type);
    dv_step = 1;
    if (_missing_mask) ms_step = _missing_mask->step, mv_step = 1;
  } else {
    dv_step = _train_data->step / CV_ELEM_SIZE(_train_data->type);
    ds_step = 1;
    if (_missing_mask) mv_step = _missing_mask->step, ms_step = 1;
  }
  tflag = _tflag;

  sample_count = sample_all;
  var_count = var_all;

  if (_sample_idx) {
    CV_CALL(sample_indices = cvPreprocessIndexArray(_sample_idx, sample_all));
    sidx = sample_indices->data.i;
    sample_count = sample_indices->rows + sample_indices->cols - 1;
  }

  if (_var_idx) {
    CV_CALL(var_idx = cvPreprocessIndexArray(_var_idx, var_all));
    vidx = var_idx->data.i;
    var_count = var_idx->rows + var_idx->cols - 1;
  }

  is_buf_16u = false;
  if (sample_count < 65536) is_buf_16u = true;

  if (!CV_IS_MAT(_responses) ||
      (CV_MAT_TYPE(_responses->type) != CV_32SC1 &&
       CV_MAT_TYPE(_responses->type) != CV_32FC1) ||
      (_responses->rows != 1 && _responses->cols != 1) ||
      _responses->rows + _responses->cols - 1 != sample_all)
    CV_ERROR(cv::Error::StsBadArg,
             "The array of _responses must be an integer or "
             "floating-point vector containing as many elements as "
             "the total number of samples in the training data matrix");

  r_type = cv::ml::VAR_CATEGORICAL;
  if (_var_type)
    CV_CALL(var_type0 =
                cvPreprocessVarType(_var_type, var_idx, var_count, &r_type));

  CV_CALL(var_type = cvCreateMat(1, var_count + 2, CV_32SC1));

  cat_var_count = 0;
  ord_var_count = -1;

  is_classifier = r_type == cv::ml::VAR_CATEGORICAL;

  // step 0. calc the number of categorical vars
  for (vi = 0; vi < var_count; vi++) {
    char vt = var_type0 ? var_type0->data.ptr[vi]
                        : static_cast<char>(cv::ml::VAR_ORDERED);
    var_type->data.i[vi] =
        vt == cv::ml::VAR_CATEGORICAL ? cat_var_count++ : ord_var_count--;
  }

  ord_var_count = ~ord_var_count;
  cv_n = params.cv_folds;
  // set the two last elements of var_type array to be able
  // to locate responses and cross-validation labels using
  // the corresponding get_* functions.
  var_type->data.i[var_count] = cat_var_count;
  var_type->data.i[var_count + 1] = cat_var_count + 1;

  // in case of single ordered predictor we need dummy cv_labels
  // for safe split_node_data() operation
  have_labels =
      cv_n > 0 || (ord_var_count == 1 && cat_var_count == 0) || _add_labels;

  work_var_count = var_count +
                   (is_classifier ? 1 : 0)   // for responses class_labels
                   + (have_labels ? 1 : 0);  // for cv_labels

  shared = _shared;
  buf_count = shared ? 2 : 1;

  buf_size = -1;  // the member buf_size is obsolete

  effective_buf_size =
      (uint64)(work_var_count + 1) * (uint64)sample_count *
      buf_count;  // this is the total size of "CvMat buf" to be allocated
  effective_buf_width = sample_count;
  effective_buf_height = work_var_count + 1;

  if (effective_buf_width >= effective_buf_height)
    effective_buf_height *= buf_count;
  else
    effective_buf_width *= buf_count;

  if ((uint64)effective_buf_width * (uint64)effective_buf_height !=
      effective_buf_size) {
    CV_Error(cv::Error::StsBadArg,
             "The memory buffer cannot be allocated since its size exceeds "
             "integer fields limit");
  }

  if (is_buf_16u) {
    CV_CALL(
        buf = cvCreateMat(effective_buf_height, effective_buf_width, CV_16UC1));
    CV_CALL(pair16u32s_ptr = (CvPair16u32s*)cvAlloc(sample_count *
                                                    sizeof(pair16u32s_ptr[0])));
  } else {
    CV_CALL(
        buf = cvCreateMat(effective_buf_height, effective_buf_width, CV_32SC1));
    CV_CALL(int_ptr = (int**)cvAlloc(sample_count * sizeof(int_ptr[0])));
  }

  size = is_classifier ? (cat_var_count + 1) : cat_var_count;
  size = !size ? 1 : size;
  CV_CALL(cat_count = cvCreateMat(1, size, CV_32SC1));
  CV_CALL(cat_ofs = cvCreateMat(1, size, CV_32SC1));

  size = is_classifier ? (cat_var_count + 1) * params.max_categories
                       : cat_var_count * params.max_categories;
  size = !size ? 1 : size;
  CV_CALL(cat_map = cvCreateMat(1, size, CV_32SC1));

  // now calculate the maximum size of split,
  // create memory storage that will keep nodes and splits of the decision tree
  // allocate root node and the buffer for the whole training data
  max_split_size = cvAlign(
      sizeof(CvDTreeSplit) + (MAX(0, sample_count - 33) / 32) * sizeof(int),
      sizeof(void*));
  tree_block_size = MAX((int)sizeof(CvDTreeNode) * 8, max_split_size);
  tree_block_size = MAX(tree_block_size + block_size_delta, min_block_size);
  CV_CALL(tree_storage = cvCreateMemStorage(tree_block_size));
  CV_CALL(node_heap = cvCreateSet(0, sizeof(*node_heap), sizeof(CvDTreeNode),
                                  tree_storage));

  nv_size = var_count * sizeof(int);
  nv_size = cvAlign(MAX(nv_size, (int)sizeof(CvSetElem)), sizeof(void*));

  temp_block_size = nv_size;

  if (cv_n) {
    if (sample_count < cv_n * MAX(params.min_sample_count, 10))
      CV_ERROR(CV_StsOutOfRange,
               "The many folds in cross-validation for such a small dataset");

    cv_size =
        cvAlign(cv_n * (sizeof(int) + sizeof(double) * 2), sizeof(double));
    temp_block_size = MAX(temp_block_size, cv_size);
  }

  temp_block_size = MAX(temp_block_size + block_size_delta, min_block_size);
  CV_CALL(temp_storage = cvCreateMemStorage(temp_block_size));
  CV_CALL(nv_heap = cvCreateSet(0, sizeof(*nv_heap), nv_size, temp_storage));
  if (cv_size)
    CV_CALL(cv_heap = cvCreateSet(0, sizeof(*cv_heap), cv_size, temp_storage));

  CV_CALL(data_root = new_node(nullptr, sample_count, 0, 0));

  max_c_count = 1;

  _fdst = nullptr;
  _idst = nullptr;
  if (ord_var_count) _fdst = (float*)cvAlloc(sample_count * sizeof(_fdst[0]));
  if (is_buf_16u && (cat_var_count || is_classifier))
    _idst = (int*)cvAlloc(sample_count * sizeof(_idst[0]));

  // transform the training data to convenient representation
  for (vi = 0; vi <= var_count; vi++) {
    int ci = 0;
    const uchar* mask = nullptr;
    int64 m_step = 0, step = 0;
    const int* idata = nullptr;
    const float* fdata = nullptr;
    int num_valid = 0;

    if (vi < var_count)  // analyze i-th input variable
    {
      int vi0 = vidx ? vidx[vi] : vi;
      ci = get_var_type(vi);
      step = ds_step;
      m_step = ms_step;
      if (CV_MAT_TYPE(_train_data->type) == CV_32SC1)
        idata = _train_data->data.i + vi0 * dv_step;
      else
        fdata = _train_data->data.fl + vi0 * dv_step;
      if (_missing_mask) mask = _missing_mask->data.ptr + vi0 * mv_step;
    } else  // analyze _responses
    {
      ci = cat_var_count;
      step = CV_IS_MAT_CONT(_responses->type)
                 ? 1
                 : _responses->step / CV_ELEM_SIZE(_responses->type);
      if (CV_MAT_TYPE(_responses->type) == CV_32SC1)
        idata = _responses->data.i;
      else
        fdata = _responses->data.fl;
    }

    if ((vi < var_count && ci >= 0) ||
        (vi == var_count &&
         is_classifier))  // process categorical variable or response
    {
      int c_count = 0, prev_label = 0;
      int* c_map = nullptr;

      if (is_buf_16u)
        udst = (unsigned short*)(buf->data.s + (size_t)vi * sample_count);
      else
        idst = buf->data.i + (size_t)vi * sample_count;

      // copy data
      for (i = 0; i < sample_count; i++) {
        int val = INT_MAX, si = sidx ? sidx[i] : i;
        if (!mask || !mask[(size_t)si * m_step]) {
          if (idata)
            val = idata[(size_t)si * step];
          else {
            float t = fdata[(size_t)si * step];
            val = cvRound(t);
            if (std::fabs(t - val) > FLT_EPSILON) {
              snprintf(err, sizeof(err),
                      "%d-th value of %d-th (categorical) "
                      "variable is not an integer",
                      i, vi);
              CV_ERROR(cv::Error::StsBadArg, err);
            }
          }

          if (val == INT_MAX) {
            snprintf(err, sizeof(err),
                    "%d-th value of %d-th (categorical) "
                    "variable is too large",
                    i, vi);
            CV_ERROR(cv::Error::StsBadArg, err);
          }
          num_valid++;
        }
        if (is_buf_16u) {
          _idst[i] = val;
          pair16u32s_ptr[i].u = udst + i;
          pair16u32s_ptr[i].i = _idst + i;
        } else {
          idst[i] = val;
          int_ptr[i] = idst + i;
        }
      }

      c_count = num_valid > 0;
      if (is_buf_16u) {
        std::sort(pair16u32s_ptr, pair16u32s_ptr + sample_count,
                  LessThanPairs());
        // count the categories
        for (i = 1; i < num_valid; i++)
          if (*pair16u32s_ptr[i].i != *pair16u32s_ptr[i - 1].i) c_count++;
      } else {
        std::sort(int_ptr, int_ptr + sample_count, LessThanPtr<int>());
        // count the categories
        for (i = 1; i < num_valid; i++)
          c_count += *int_ptr[i] != *int_ptr[i - 1];
      }

      if (vi > 0) max_c_count = MAX(max_c_count, c_count);
      cat_count->data.i[ci] = c_count;
      cat_ofs->data.i[ci] = total_c_count;

      // resize cat_map, if need
      if (cat_map->cols < total_c_count + c_count) {
        tmp_map = cat_map;
        CV_CALL(cat_map = cvCreateMat(
                    1, MAX(cat_map->cols * 3 / 2, total_c_count + c_count),
                    CV_32SC1));
        for (i = 0; i < total_c_count; i++)
          cat_map->data.i[i] = tmp_map->data.i[i];
        cvReleaseMat(&tmp_map);
      }

      c_map = cat_map->data.i + total_c_count;
      total_c_count += c_count;

      c_count = -1;
      if (is_buf_16u) {
        // compact the class indices and build the map
        prev_label = ~*pair16u32s_ptr[0].i;
        for (i = 0; i < num_valid; i++) {
          int cur_label = *pair16u32s_ptr[i].i;
          if (cur_label != prev_label)
            c_map[++c_count] = prev_label = cur_label;
          *pair16u32s_ptr[i].u = (unsigned short)c_count;
        }
        // replace labels for missing values with -1
        for (; i < sample_count; i++) *pair16u32s_ptr[i].u = 65535;
      } else {
        // compact the class indices and build the map
        prev_label = ~*int_ptr[0];
        for (i = 0; i < num_valid; i++) {
          int cur_label = *int_ptr[i];
          if (cur_label != prev_label)
            c_map[++c_count] = prev_label = cur_label;
          *int_ptr[i] = c_count;
        }
        // replace labels for missing values with -1
        for (; i < sample_count; i++) *int_ptr[i] = -1;
      }
    } else if (ci < 0)  // process ordered variable
    {
      if (is_buf_16u)
        udst = (unsigned short*)(buf->data.s + (size_t)vi * sample_count);
      else
        idst = buf->data.i + (size_t)vi * sample_count;

      for (i = 0; i < sample_count; i++) {
        float val = ord_nan;
        int si = sidx ? sidx[i] : i;
        if (!mask || !mask[(size_t)si * m_step]) {
          if (idata)
            val = (float)idata[(size_t)si * step];
          else
            val = fdata[(size_t)si * step];

          if (std::fabs(val) >= ord_nan) {
            snprintf(err, sizeof(err),
                    "%d-th value of %d-th (ordered) "
                    "variable (=%g) is too large",
                    i, vi, val);
            CV_ERROR(cv::Error::StsBadArg, err);
          }
          num_valid++;
        }

        if (is_buf_16u)
          udst[i] = (unsigned short)i;  // TODO: memory corruption may be here
        else
          idst[i] = i;
        _fdst[i] = val;
      }
      if (is_buf_16u)
        std::sort(udst, udst + sample_count,
                  LessThanIdx<float, unsigned short>(_fdst));
      else
        std::sort(idst, idst + sample_count, LessThanIdx<float, int>(_fdst));
    }

    if (vi < var_count) data_root->set_num_valid(vi, num_valid);
  }

  // set sample labels
  if (is_buf_16u)
    udst =
        (unsigned short*)(buf->data.s + (size_t)work_var_count * sample_count);
  else
    idst = buf->data.i + (size_t)work_var_count * sample_count;

  for (i = 0; i < sample_count; i++) {
    if (udst)
      udst[i] = sidx ? (unsigned short)sidx[i] : (unsigned short)i;
    else
      idst[i] = sidx ? sidx[i] : i;
  }

  if (cv_n) {
    unsigned short* usdst = nullptr;
    int* idst2 = nullptr;

    if (is_buf_16u) {
      usdst =
          (unsigned short*)(buf->data.s +
                            (size_t)(get_work_var_count() - 1) * sample_count);
      for (i = vi = 0; i < sample_count; i++) {
        usdst[i] = (unsigned short)vi++;
        vi &= vi < cv_n ? -1 : 0;
      }

      for (i = 0; i < sample_count; i++) {
        int a = (*rng)(sample_count);
        int b = (*rng)(sample_count);
        auto unsh = (unsigned short)vi;
        CV_SWAP(usdst[a], usdst[b], unsh);
      }
    } else {
      idst2 = buf->data.i + (size_t)(get_work_var_count() - 1) * sample_count;
      for (i = vi = 0; i < sample_count; i++) {
        idst2[i] = vi++;
        vi &= vi < cv_n ? -1 : 0;
      }

      for (i = 0; i < sample_count; i++) {
        int a = (*rng)(sample_count);
        int b = (*rng)(sample_count);
        CV_SWAP(idst2[a], idst2[b], vi);
      }
    }
  }

  if (cat_map) cat_map->cols = MAX(total_c_count, 1);

  max_split_size = cvAlign(
      sizeof(CvDTreeSplit) + (MAX(0, max_c_count - 33) / 32) * sizeof(int),
      sizeof(void*));
  CV_CALL(split_heap = cvCreateSet(0, sizeof(*split_heap), max_split_size,
                                   tree_storage));

  have_priors = is_classifier && params.priors;
  if (is_classifier) {
    int m = get_num_classes();
    double sum = 0;
    CV_CALL(priors = cvCreateMat(1, m, CV_64F));
    for (i = 0; i < m; i++) {
      double val = have_priors ? params.priors[i] : 1.;
      if (val <= 0)
        CV_ERROR(CV_StsOutOfRange, "Every class weight should be positive");
      priors->data.db[i] = val;
      sum += val;
    }

    // normalize weights
    if (have_priors) cvScale(priors, priors, 1. / sum);

    CV_CALL(priors_mult = cvCloneMat(priors));
    CV_CALL(counts = cvCreateMat(1, m, CV_32SC1));
  }

  CV_CALL(direction = cvCreateMat(1, sample_count, CV_8UC1));
  CV_CALL(split_buf = cvCreateMat(1, sample_count, CV_32SC1));

  __CV_END__;

  if (data) delete data;

  if (_fdst) cvFree(&_fdst);
  if (_idst) cvFree(&_idst);
  cvFree(&int_ptr);
  cvFree(&pair16u32s_ptr);
  cvReleaseMat(&var_type0);
  cvReleaseMat(&sample_indices);
  cvReleaseMat(&tmp_map);
}

void CvDTreeTrainData::do_responses_copy() {
  responses_copy =
      cvCreateMat(responses->rows, responses->cols, responses->type);
  cvCopy(responses, responses_copy);
  responses = responses_copy;
}

CvDTreeNode* CvDTreeTrainData::subsample_data(const CvMat* _subsample_idx) {
  CvDTreeNode* root = nullptr;
  CvMat* isubsample_idx = nullptr;
  CvMat* subsample_co = nullptr;

  bool isMakeRootCopy = true;

  CV_FUNCNAME("CvDTreeTrainData::subsample_data");

  __CV_BEGIN__;

  if (!data_root) CV_ERROR(CV_StsError, "No training data has been set");

  if (_subsample_idx) {
    CV_CALL(isubsample_idx =
                cvPreprocessIndexArray(_subsample_idx, sample_count));

    if (isubsample_idx->cols + isubsample_idx->rows - 1 == sample_count) {
      const int* sidx = isubsample_idx->data.i;
      for (int i = 0; i < sample_count; i++) {
        if (sidx[i] != i) {
          isMakeRootCopy = false;
          break;
        }
      }
    } else
      isMakeRootCopy = false;
  }

  if (isMakeRootCopy) {
    // make a copy of the root node
    CvDTreeNode temp{};
    int i = 0;
    root = new_node(nullptr, 1, 0, 0);
    temp = *root;
    *root = *data_root;
    root->num_valid = temp.num_valid;
    if (root->num_valid) {
      for (i = 0; i < var_count; i++)
        root->num_valid[i] = data_root->num_valid[i];
    }
    root->cv_Tn = temp.cv_Tn;
    root->cv_node_risk = temp.cv_node_risk;
    root->cv_node_error = temp.cv_node_error;
  } else {
    int* sidx = isubsample_idx->data.i;
    // co - array of count/offset pairs (to handle duplicated values in
    // _subsample_idx)
    int *co = nullptr, cur_ofs = 0;
    int vi = 0, i = 0;
    int workVarCount = get_work_var_count();
    int count = isubsample_idx->rows + isubsample_idx->cols - 1;

    root = new_node(nullptr, count, 1, 0);

    CV_CALL(subsample_co = cvCreateMat(1, sample_count * 2, CV_32SC1));
    cvZero(subsample_co);
    co = subsample_co->data.i;
    for (i = 0; i < count; i++) co[sidx[i] * 2]++;
    for (i = 0; i < sample_count; i++) {
      if (co[i * 2]) {
        co[i * 2 + 1] = cur_ofs;
        cur_ofs += co[i * 2];
      } else
        co[i * 2 + 1] = -1;
    }

    cv::AutoBuffer<uchar> inn_buf(sample_count *
                                  (2 * sizeof(int) + sizeof(float)));
    for (vi = 0; vi < workVarCount; vi++) {
      int ci = get_var_type(vi);

      if (ci >= 0 || vi >= var_count) {
        int num_valid = 0;
        const int* src = CvDTreeTrainData::get_cat_var_data(
            data_root, vi, (int*)inn_buf.data());

        if (is_buf_16u) {
          auto* udst =
              (unsigned short*)(buf->data.s +
                                root->buf_idx * get_length_subbuf() +
                                (size_t)vi * sample_count + root->offset);
          for (i = 0; i < count; i++) {
            int val = src[sidx[i]];
            udst[i] = (unsigned short)val;
            num_valid += val >= 0;
          }
        } else {
          int* idst = buf->data.i + root->buf_idx * get_length_subbuf() +
                      (size_t)vi * sample_count + root->offset;
          for (i = 0; i < count; i++) {
            int val = src[sidx[i]];
            idst[i] = val;
            num_valid += val >= 0;
          }
        }

        if (vi < var_count) root->set_num_valid(vi, num_valid);
      } else {
        int* src_idx_buf = (int*)inn_buf.data();
        auto* src_val_buf = (float*)(src_idx_buf + sample_count);
        int* sample_indices_buf = (int*)(src_val_buf + sample_count);
        const int* src_idx = nullptr;
        const float* src_val = nullptr;
        get_ord_var_data(data_root, vi, src_val_buf, src_idx_buf, &src_val,
                         &src_idx, sample_indices_buf);
        int j = 0, idx = 0, count_i = 0;
        int num_valid = data_root->get_num_valid(vi);

        if (is_buf_16u) {
          auto* udst_idx =
              (unsigned short*)(buf->data.s +
                                root->buf_idx * get_length_subbuf() +
                                (size_t)vi * sample_count + data_root->offset);
          for (i = 0; i < num_valid; i++) {
            idx = src_idx[i];
            count_i = co[idx * 2];
            if (count_i)
              for (cur_ofs = co[idx * 2 + 1]; count_i > 0;
                   count_i--, j++, cur_ofs++)
                udst_idx[j] = (unsigned short)cur_ofs;
          }

          root->set_num_valid(vi, j);

          for (; i < sample_count; i++) {
            idx = src_idx[i];
            count_i = co[idx * 2];
            if (count_i)
              for (cur_ofs = co[idx * 2 + 1]; count_i > 0;
                   count_i--, j++, cur_ofs++)
                udst_idx[j] = (unsigned short)cur_ofs;
          }
        } else {
          int* idst_idx = buf->data.i + root->buf_idx * get_length_subbuf() +
                          (size_t)vi * sample_count + root->offset;
          for (i = 0; i < num_valid; i++) {
            idx = src_idx[i];
            count_i = co[idx * 2];
            if (count_i)
              for (cur_ofs = co[idx * 2 + 1]; count_i > 0;
                   count_i--, j++, cur_ofs++)
                idst_idx[j] = cur_ofs;
          }

          root->set_num_valid(vi, j);

          for (; i < sample_count; i++) {
            idx = src_idx[i];
            count_i = co[idx * 2];
            if (count_i)
              for (cur_ofs = co[idx * 2 + 1]; count_i > 0;
                   count_i--, j++, cur_ofs++)
                idst_idx[j] = cur_ofs;
          }
        }
      }
    }
    // sample indices subsampling
    const int* sample_idx_src =
        get_sample_indices(data_root, (int*)inn_buf.data());
    if (is_buf_16u) {
      auto* sample_idx_dst =
          (unsigned short*)(buf->data.s + root->buf_idx * get_length_subbuf() +
                            (size_t)workVarCount * sample_count + root->offset);
      for (i = 0; i < count; i++)
        sample_idx_dst[i] = (unsigned short)sample_idx_src[sidx[i]];
    } else {
      int* sample_idx_dst = buf->data.i + root->buf_idx * get_length_subbuf() +
                            (size_t)workVarCount * sample_count + root->offset;
      for (i = 0; i < count; i++) sample_idx_dst[i] = sample_idx_src[sidx[i]];
    }
  }

  __CV_END__;

  cvReleaseMat(&isubsample_idx);
  cvReleaseMat(&subsample_co);

  return root;
}

void CvDTreeTrainData::get_vectors(const CvMat* _subsample_idx, float* values,
                                   uchar* missing, float* _responses,
                                   bool get_class_idx) {
  CvMat* subsample_idx = nullptr;
  CvMat* subsample_co = nullptr;

  CV_FUNCNAME("CvDTreeTrainData::get_vectors");

  __CV_BEGIN__;

  int i = 0, vi = 0, total = sample_count, count = total, cur_ofs = 0;
  int* sidx = nullptr;
  int* co = nullptr;

  cv::AutoBuffer<uchar> inn_buf(sample_count *
                                (2 * sizeof(int) + sizeof(float)));
  if (_subsample_idx) {
    CV_CALL(subsample_idx =
                cvPreprocessIndexArray(_subsample_idx, sample_count));
    sidx = subsample_idx->data.i;
    CV_CALL(subsample_co = cvCreateMat(1, sample_count * 2, CV_32SC1));
    co = subsample_co->data.i;
    cvZero(subsample_co);
    count = subsample_idx->cols + subsample_idx->rows - 1;
    for (i = 0; i < count; i++) co[sidx[i] * 2]++;
    for (i = 0; i < total; i++) {
      int count_i = co[i * 2];
      if (count_i) {
        co[i * 2 + 1] = cur_ofs * var_count;
        cur_ofs += count_i;
      }
    }
  }

  if (missing) memset(missing, 1, (size_t)count * var_count);

  for (vi = 0; vi < var_count; vi++) {
    int ci = get_var_type(vi);
    if (ci >= 0)  // categorical
    {
      float* dst = values + vi;
      uchar* m = missing ? missing + vi : nullptr;
      const int* src = get_cat_var_data(data_root, vi, (int*)inn_buf.data());

      for (i = 0; i < count; i++, dst += var_count) {
        int idx = sidx ? sidx[i] : i;
        int val = src[idx];
        *dst = (float)val;
        if (m) {
          *m = (!is_buf_16u && val < 0) || (is_buf_16u && (val == 65535));
          m += var_count;
        }
      }
    } else  // ordered
    {
      float* dst = values + vi;
      uchar* m = missing ? missing + vi : nullptr;
      int count1 = data_root->get_num_valid(vi);
      auto* src_val_buf = (float*)inn_buf.data();
      int* src_idx_buf = (int*)(src_val_buf + sample_count);
      int* sample_indices_buf = src_idx_buf + sample_count;
      const float* src_val = nullptr;
      const int* src_idx = nullptr;
      get_ord_var_data(data_root, vi, src_val_buf, src_idx_buf, &src_val,
                       &src_idx, sample_indices_buf);

      for (i = 0; i < count1; i++) {
        int idx = src_idx[i];
        int count_i = 1;
        if (co) {
          count_i = co[idx * 2];
          cur_ofs = co[idx * 2 + 1];
        } else
          cur_ofs = idx * var_count;
        if (count_i) {
          float val = src_val[i];
          for (; count_i > 0; count_i--, cur_ofs += var_count) {
            dst[cur_ofs] = val;
            if (m) m[cur_ofs] = 0;
          }
        }
      }
    }
  }

  // copy responses
  if (_responses) {
    if (is_classifier) {
      const int* src = get_class_labels(data_root, (int*)inn_buf.data());
      for (i = 0; i < count; i++) {
        int idx = sidx ? sidx[i] : i;
        int val =
            get_class_idx
                ? src[idx]
                : cat_map->data.i[cat_ofs->data.i[cat_var_count] + src[idx]];
        _responses[i] = (float)val;
      }
    } else {
      auto* val_buf = (float*)inn_buf.data();
      int* sample_idx_buf = (int*)(val_buf + sample_count);
      const float* _values =
          get_ord_responses(data_root, val_buf, sample_idx_buf);
      for (i = 0; i < count; i++) {
        int idx = sidx ? sidx[i] : i;
        _responses[i] = _values[idx];
      }
    }
  }

  __CV_END__;

  cvReleaseMat(&subsample_idx);
  cvReleaseMat(&subsample_co);
}

CvDTreeNode* CvDTreeTrainData::new_node(CvDTreeNode* parent, int count,
                                        int storage_idx, int offset) {
  auto* node = (CvDTreeNode*)cvSetNew(node_heap);

  node->sample_count = count;
  node->depth = parent ? parent->depth + 1 : 0;
  node->parent = parent;
  node->left = node->right = nullptr;
  node->split = nullptr;
  node->value = 0;
  node->class_idx = 0;
  node->maxlr = 0.;

  node->buf_idx = storage_idx;
  node->offset = offset;
  if (nv_heap)
    node->num_valid = (int*)cvSetNew(nv_heap);
  else
    node->num_valid = nullptr;
  node->alpha = node->node_risk = node->tree_risk = node->tree_error = 0.;
  node->complexity = 0;

  if (params.cv_folds > 0 && cv_heap) {
    int cv_n = params.cv_folds;
    node->Tn = INT_MAX;
    node->cv_Tn = (int*)cvSetNew(cv_heap);
    node->cv_node_risk =
        (double*)cv::alignPtr(node->cv_Tn + cv_n, sizeof(double));
    node->cv_node_error = node->cv_node_risk + cv_n;
  } else {
    node->Tn = 0;
    node->cv_Tn = nullptr;
    node->cv_node_risk = nullptr;
    node->cv_node_error = nullptr;
  }

  return node;
}

CvDTreeSplit* CvDTreeTrainData::new_split_ord(int vi, float cmp_val,
                                              int split_point, int inversed,
                                              float quality) {
  auto* split = (CvDTreeSplit*)cvSetNew(split_heap);
  split->var_idx = vi;
  split->condensed_idx = INT_MIN;
  split->ord.c = cmp_val;
  split->ord.split_point = split_point;
  split->inversed = inversed;
  split->quality = quality;
  split->next = nullptr;

  return split;
}

CvDTreeSplit* CvDTreeTrainData::new_split_cat(int vi, float quality) {
  auto* split = (CvDTreeSplit*)cvSetNew(split_heap);
  int i = 0, n = (max_c_count + 31) / 32;

  split->var_idx = vi;
  split->condensed_idx = INT_MIN;
  split->inversed = 0;
  split->quality = quality;
  for (i = 0; i < n; i++) split->subset[i] = 0;
  split->next = nullptr;

  return split;
}

void CvDTreeTrainData::free_node(CvDTreeNode* node) {
  CvDTreeSplit* split = node->split;
  free_node_data(node);
  while (split) {
    CvDTreeSplit* next = split->next;
    cvSetRemoveByPtr(split_heap, split);
    split = next;
  }
  node->split = nullptr;
  cvSetRemoveByPtr(node_heap, node);
}

void CvDTreeTrainData::free_node_data(CvDTreeNode* node) {
  if (node->num_valid) {
    cvSetRemoveByPtr(nv_heap, node->num_valid);
    node->num_valid = nullptr;
  }
  // do not free cv_* fields, as all the cross-validation related data is
  // released at once.
}

void CvDTreeTrainData::free_train_data() {
  cvReleaseMat(&counts);
  cvReleaseMat(&buf);
  cvReleaseMat(&direction);
  cvReleaseMat(&split_buf);
  cvReleaseMemStorage(&temp_storage);
  cvReleaseMat(&responses_copy);
  cv_heap = nv_heap = nullptr;
}

void CvDTreeTrainData::clear() {
  free_train_data();

  cvReleaseMemStorage(&tree_storage);

  cvReleaseMat(&var_idx);
  cvReleaseMat(&var_type);
  cvReleaseMat(&cat_count);
  cvReleaseMat(&cat_ofs);
  cvReleaseMat(&cat_map);
  cvReleaseMat(&priors);
  cvReleaseMat(&priors_mult);

  node_heap = split_heap = nullptr;

  sample_count = var_all = var_count = max_c_count = ord_var_count =
      cat_var_count = 0;
  have_labels = have_priors = is_classifier = false;

  buf_count = buf_size = 0;
  shared = false;

  data_root = nullptr;

  rng = &cv::theRNG();
}

int CvDTreeTrainData::get_num_classes() const {
  return is_classifier ? cat_count->data.i[cat_var_count] : 0;
}

int CvDTreeTrainData::get_var_type(int vi) const {
  return var_type->data.i[vi];
}

void CvDTreeTrainData::get_ord_var_data(CvDTreeNode* n, int vi,
                                        float* ord_values_buf,
                                        int* sorted_indices_buf,
                                        const float** ord_values,
                                        const int** sorted_indices,
                                        int* sample_indices_buf) {
  int vidx = var_idx ? var_idx->data.i[vi] : vi;
  int node_sample_count = n->sample_count;
  int td_step = train_data->step / CV_ELEM_SIZE(train_data->type);

  const int* sample_indices = get_sample_indices(n, sample_indices_buf);

  if (!is_buf_16u)
    *sorted_indices = buf->data.i + n->buf_idx * get_length_subbuf() +
                      (size_t)vi * sample_count + n->offset;
  else {
    const auto* short_indices =
        (const unsigned short*)(buf->data.s + n->buf_idx * get_length_subbuf() +
                                (size_t)vi * sample_count + n->offset);
    for (int i = 0; i < node_sample_count; i++)
      sorted_indices_buf[i] = short_indices[i];
    *sorted_indices = sorted_indices_buf;
  }

  if (tflag == cv::ml::ROW_SAMPLE) {
    for (int i = 0; i < node_sample_count &&
                    ((((*sorted_indices)[i] >= 0) && !is_buf_16u) ||
                     (((*sorted_indices)[i] != 65535) && is_buf_16u));
         i++) {
      int idx = (*sorted_indices)[i];
      idx = sample_indices[idx];
      ord_values_buf[i] = *(train_data->data.fl + idx * td_step + vidx);
    }
  } else
    for (int i = 0; i < node_sample_count &&
                    ((((*sorted_indices)[i] >= 0) && !is_buf_16u) ||
                     (((*sorted_indices)[i] != 65535) && is_buf_16u));
         i++) {
      int idx = (*sorted_indices)[i];
      idx = sample_indices[idx];
      ord_values_buf[i] = *(train_data->data.fl + vidx * td_step + idx);
    }

  *ord_values = ord_values_buf;
}

const int* CvDTreeTrainData::get_class_labels(CvDTreeNode* n, int* labels_buf) {
  if (is_classifier) return get_cat_var_data(n, var_count, labels_buf);
  return nullptr;
}

const int* CvDTreeTrainData::get_sample_indices(CvDTreeNode* n,
                                                int* indices_buf) {
  return get_cat_var_data(n, get_work_var_count(), indices_buf);
}

const float* CvDTreeTrainData::get_ord_responses(CvDTreeNode* n,
                                                 float* values_buf,
                                                 int* sample_indices_buf) {
  int _sample_count = n->sample_count;
  int r_step = CV_IS_MAT_CONT(responses->type)
                   ? 1
                   : responses->step / CV_ELEM_SIZE(responses->type);
  const int* indices = get_sample_indices(n, sample_indices_buf);

  for (int i = 0; i < _sample_count && (((indices[i] >= 0) && !is_buf_16u) ||
                                        ((indices[i] != 65535) && is_buf_16u));
       i++) {
    int idx = indices[i];
    values_buf[i] = *(responses->data.fl + idx * r_step);
  }

  return values_buf;
}

const int* CvDTreeTrainData::get_cv_labels(CvDTreeNode* n, int* labels_buf) {
  if (have_labels)
    return get_cat_var_data(n, get_work_var_count() - 1, labels_buf);
  return nullptr;
}

const int* CvDTreeTrainData::get_cat_var_data(CvDTreeNode* n, int vi,
                                              int* cat_values_buf) {
  const int* cat_values = nullptr;
  if (!is_buf_16u)
    cat_values = buf->data.i + n->buf_idx * get_length_subbuf() +
                 (size_t)vi * sample_count + n->offset;
  else {
    const auto* short_values =
        (const unsigned short*)(buf->data.s + n->buf_idx * get_length_subbuf() +
                                (size_t)vi * sample_count + n->offset);
    for (int i = 0; i < n->sample_count; i++)
      cat_values_buf[i] = short_values[i];
    cat_values = cat_values_buf;
  }
  return cat_values;
}

int CvDTreeTrainData::get_child_buf_idx(CvDTreeNode* n) {
  int idx = n->buf_idx + 1;
  if (idx >= buf_count) idx = shared ? 1 : 0;
  return idx;
}
