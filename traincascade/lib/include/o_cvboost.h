/**
 * @file o_cvboost.h
 * @brief Legacy OpenCV @c CvBoost: AdaBoost ensemble of decision trees.
 *
 * This is the canonical boosting machinery the cascade trainer extends.
 * It maintains the per-sample weight vector, holds the ordered list of
 * weak learners (decision trees implemented by
 * @ref CvBoostTree), and offers the standard predict/clear/getters
 * exposed by every legacy @c CvStatModel.
 */
#pragma once

#include <opencv2/core/core_c.h>

#include "o_cvboostparams.h"
#include "o_cvstatmodel.h"

struct CvDTreeTrainData;

/**
 * @brief AdaBoost ensemble of small decision trees.
 *
 * Used unchanged inside the cascade as @ref CvCascadeBoost's base class.
 * Configuration is passed via @ref CvBoostParams (boost variant, weak
 * count, weight trim rate, ...). Concrete weak learners are
 * @ref CvBoostTree instances stored in the @c weak sequence.
 */
class CvBoost : public CvStatModel {
 public:
  /// Boosting variant used during training.
  enum { DISCRETE = 0, REAL = 1, LOGIT = 2, GENTLE = 3 };

  /// Per-split scoring criterion used by individual weak trees.
  enum { DEFAULT = 0, GINI = 1, MISCLASS = 3, SQERR = 4 };

  CvBoost();
  virtual ~CvBoost();

  /**
   * @brief One-shot constructor: build a model and immediately train it
   *        on the supplied data set.
   * @see CvDTree::train for the meaning of the data-related parameters.
   */
  CvBoost(const CvMat* trainData, int tflag, const CvMat* responses,
          const CvMat* varIdx = 0, const CvMat* sampleIdx = 0,
          const CvMat* varType = 0, const CvMat* missingDataMask = 0,
          CvBoostParams params = CvBoostParams());

  /// Drop weak learners whose indices fall in @p slice.
  virtual void prune(CvSlice slice);

  /// Release every internal buffer (weights, weak learners, working data).
  void clear() override;

  /// Return the (typed) sequence of weak learners.
  CvSeq* get_weak_predictors();

  CvMat* get_weights();
  CvMat* get_subtree_weights();
  CvMat* get_weak_response();
  const CvBoostParams& get_params() const;

 protected:
  /// Validate and copy parameters; called by every public train overload.
  virtual bool set_params(const CvBoostParams& params);
  /// Apply weight trimming so very-low-weight samples are ignored next round.
  virtual void trim_weights();

  CvDTreeTrainData* data;
  CvMat train_data_hdr{}, responses_hdr{};
  cv::Mat train_data_mat, responses_mat;
  CvBoostParams params;
  CvSeq* weak; ///< Trained weak learners in order of evaluation.

  CvMat* active_vars;
  CvMat* active_vars_abs;
  bool have_active_cat_vars;

  CvMat* orig_response;
  CvMat* sum_response;
  CvMat* weak_eval;
  CvMat* subsample_mask;
  CvMat* weights;          ///< Current per-sample boosting weights.
  CvMat* subtree_weights;
  bool have_subsample;
};
