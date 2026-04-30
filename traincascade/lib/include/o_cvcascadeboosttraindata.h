/**
 * @file o_cvcascadeboosttraindata.h
 * @brief Specialization of @ref CvDTreeTrainData for cascade boosting.
 *
 * The cascade trainer computes feature responses lazily through a
 * @ref CvFeatureEvaluator. Building a full numeric @c trainData matrix
 * up front would be prohibitive (millions of features per sample), so
 * @ref CvCascadeBoostTrainData precomputes only a small number of the
 * highest-ranked feature columns (@c numPrecalcVal feature values plus
 * @c numPrecalcIdx sorted-index columns) and falls back to @ref getVarValue
 * for everything else.
 */
#pragma once

#include <opencv2/core/types_c.h>

#include "o_cvdtreeparams.h"
#include "o_cvdtreetraindata.h"

class CvFeatureEvaluator;
struct CvDTreeNode;

/**
 * @brief Train-data store used by @ref CvCascadeBoost.
 *
 * Overrides the @c get_*_data accessors so the boosting trainer can
 * iterate features that are not in the precalculated cache transparently.
 * @c valCache holds the precalculated feature values for the
 * @c numPrecalcVal best features (selected by variance).
 */
struct CvCascadeBoostTrainData : CvDTreeTrainData {
  CvCascadeBoostTrainData(const CvFeatureEvaluator* _featureEvaluator,
                          const CvDTreeParams& _params);
  CvCascadeBoostTrainData(const CvFeatureEvaluator* _featureEvaluator,
                          int _numSamples, int _precalcValBufSize,
                          int _precalcIdxBufSize,
                          const CvDTreeParams& _params = CvDTreeParams());
  /// Reconfigure the data store; reallocates @c valCache.
  virtual void setData(const CvFeatureEvaluator* _featureEvaluator,
                       int _numSamples, int _precalcValBufSize,
                       int _precalcIdxBufSize,
                       const CvDTreeParams& _params = CvDTreeParams());
  /// Populate @c valCache with the @c numPrecalcVal highest-variance feature columns.
  void precalculate();

  CvDTreeNode* subsample_data(const CvMat* _subsample_idx) override;

  const int* get_class_labels(CvDTreeNode* n, int* labelsBuf) override;
  const int* get_cv_labels(CvDTreeNode* n, int* labelsBuf) override;
  const int* get_sample_indices(CvDTreeNode* n, int* indicesBuf) override;

  void get_ord_var_data(CvDTreeNode* n, int vi, float* ordValuesBuf,
                        int* sortedIndicesBuf, const float** ordValues,
                        const int** sortedIndices,
                        int* sampleIndicesBuf) override;
  const int* get_cat_var_data(CvDTreeNode* n, int vi,
                              int* catValuesBuf) override;
  /// Compute (or look up in @c valCache) the value of feature @p vi on sample @p si.
  virtual float getVarValue(int vi, int si);
  void free_train_data() override;

  const CvFeatureEvaluator* featureEvaluator{}; ///< Non-owning evaluator used to compute feature values on demand.
  cv::Mat valCache;  // precalculated feature values (CV_32FC1)
  CvMat _resp{};       // for casting
  int numPrecalcVal{}, numPrecalcIdx{};
};
