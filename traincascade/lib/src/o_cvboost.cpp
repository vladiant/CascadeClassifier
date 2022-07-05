#include "o_cvboost.h"

#include <opencv2/ml/ml.hpp>

#include "o_cvboostree.h"
#include "o_cvdtreeparams.h"
#include "o_cvdtreetraindata.h"

CvBoost::CvBoost() {
  data = nullptr;
  weak = nullptr;
  default_model_name = "my_boost_tree";

  active_vars = active_vars_abs = orig_response = sum_response = weak_eval =
      subsample_mask = weights = subtree_weights = 0;
  have_active_cat_vars = have_subsample = false;

  clear();
}

void CvBoost::prune(CvSlice slice) {
  if (weak && weak->total > 0) {
    CvSeqReader reader;
    int i, count = cvSliceLength(slice, weak);

    cvStartReadSeq(weak, &reader);
    cvSetSeqReaderPos(&reader, slice.start_index);

    static_cast<void>(i);
    static_cast<void>(count);
    for (i = 0; i < count; i++) {
      CvBoostTree* w;
      CV_READ_SEQ_ELEM(w, reader);
      delete w;
    }

    cvSeqRemoveSlice(weak, slice);
  }
}

void CvBoost::clear() {
  if (weak) {
    prune(CV_WHOLE_SEQ);
    cvReleaseMemStorage(&weak->storage);
  }
  if (data) delete data;
  weak = nullptr;
  data = nullptr;
  cvReleaseMat(&active_vars);
  cvReleaseMat(&active_vars_abs);
  cvReleaseMat(&orig_response);
  cvReleaseMat(&sum_response);
  cvReleaseMat(&weak_eval);
  cvReleaseMat(&subsample_mask);
  cvReleaseMat(&weights);
  cvReleaseMat(&subtree_weights);

  have_subsample = false;
}

CvBoost::~CvBoost() { clear(); }

CvSeq* CvBoost::get_weak_predictors() { return weak; }

bool CvBoost::set_params(const CvBoostParams& _params) {
  bool ok = false;

  CV_FUNCNAME("CvBoost::set_params");

  __CV_BEGIN__;

  params = _params;
  if (params.boost_type != DISCRETE && params.boost_type != REAL &&
      params.boost_type != LOGIT && params.boost_type != GENTLE)
    CV_ERROR(cv::Error::StsBadArg, "Unknown/unsupported boosting type");

  params.weak_count = MAX(params.weak_count, 1);
  params.weight_trim_rate = MAX(params.weight_trim_rate, 0.);
  params.weight_trim_rate = MIN(params.weight_trim_rate, 1.);
  if (params.weight_trim_rate < FLT_EPSILON) params.weight_trim_rate = 1.f;

  if (params.boost_type == DISCRETE && params.split_criteria != GINI &&
      params.split_criteria != MISCLASS)
    params.split_criteria = MISCLASS;
  if (params.boost_type == REAL && params.split_criteria != GINI &&
      params.split_criteria != MISCLASS)
    params.split_criteria = GINI;
  if ((params.boost_type == LOGIT || params.boost_type == GENTLE) &&
      params.split_criteria != SQERR)
    params.split_criteria = SQERR;

  ok = true;

  __CV_END__;

  return ok;
}

void CvBoost::trim_weights() {
  // CV_FUNCNAME( "CvBoost::trim_weights" );

  __CV_BEGIN__;

  int i, count = data->sample_count, nz_count = 0;
  double sum, threshold;

  if (params.weight_trim_rate <= 0. || params.weight_trim_rate >= 1.)
    __CV_EXIT__;

  // use weak_eval as temporary buffer for sorted weights
  cvCopy(weights, weak_eval);

  std::sort(weak_eval->data.db, weak_eval->data.db + count);

  // as weight trimming occurs immediately after updating the weights,
  // where they are renormalized, we assume that the weight sum = 1.
  sum = 1. - params.weight_trim_rate;

  for (i = 0; i < count; i++) {
    double w = weak_eval->data.db[i];
    if (sum <= 0) break;
    sum -= w;
  }

  threshold = i < count ? weak_eval->data.db[i] : DBL_MAX;

  for (i = 0; i < count; i++) {
    double w = weights->data.db[i];
    int f = w >= threshold;
    subsample_mask->data.ptr[i] = (uchar)f;
    nz_count += f;
  }

  have_subsample = nz_count < count;

  __CV_END__;
}

CvMat* CvBoost::get_weak_response() { return weak_eval; }

CvMat* CvBoost::get_subtree_weights() { return subtree_weights; }

const CvBoostParams& CvBoost::get_params() const { return params; }

CvMat* CvBoost::get_weights() { return weights; }