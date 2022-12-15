#pragma once

#include <opencv2/core/types_c.h>

#include "o_cvdtreeparams.h"
#include "o_cvdtreetraindata.h"

class CvFeatureEvaluator;
struct CvDTreeNode;

// CvCascadeBoostTree
struct CvCascadeBoostTrainData : CvDTreeTrainData {
  CvCascadeBoostTrainData(const CvFeatureEvaluator* _featureEvaluator,
                          const CvDTreeParams& _params);
  CvCascadeBoostTrainData(const CvFeatureEvaluator* _featureEvaluator,
                          int _numSamples, int _precalcValBufSize,
                          int _precalcIdxBufSize,
                          const CvDTreeParams& _params = CvDTreeParams());
  virtual void setData(const CvFeatureEvaluator* _featureEvaluator,
                       int _numSamples, int _precalcValBufSize,
                       int _precalcIdxBufSize,
                       const CvDTreeParams& _params = CvDTreeParams());
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
  virtual float getVarValue(int vi, int si);
  void free_train_data() override;

  const CvFeatureEvaluator* featureEvaluator{};
  cv::Mat valCache;  // precalculated feature values (CV_32FC1)
  CvMat _resp{};       // for casting
  int numPrecalcVal{}, numPrecalcIdx{};
};
