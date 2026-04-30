/**
 * @file o_cvcascadeboosttree.h
 * @brief Weak tree subclass tailored for cascade stages.
 */
#pragma once

#include "o_cvboostree.h"

struct CvDTreeNode;
class CvBoost;
struct CvDTreeTrainData;

namespace cv {
class FileStorage;
class FileNode;
class Mat;
}  // namespace cv

/**
 * @brief Decision tree weak learner specialized for cascade boosting.
 *
 * Adds an integer-indexed @ref predict overload that consults the
 * cascade-specific @ref CvCascadeBoostTrainData (instead of going
 * through CvMat sample objects) and a @ref write / @ref read pair that
 * uses the cascade XML schema. @ref split_node_data is overridden so
 * the child sample assignments stay consistent with the cached feature
 * value matrix maintained by @ref CvCascadeBoostTrainData.
 */
class CvCascadeBoostTree : public CvBoostTree {
 public:
  using CvBoostTree::predict;
  /// Predict the leaf reached by sample @p sampleIdx using cached data.
  CvDTreeNode* predict(int sampleIdx) const;
  /// Serialize the tree using the cascade XML schema; @p featureMap
  /// remaps used feature indices to compact ones.
  void write(cv::FileStorage& fs, const cv::Mat& featureMap);
  /// Restore a previously-saved tree.
  void read(const cv::FileNode& node, CvBoost* _ensemble,
            CvDTreeTrainData* _data);
  /// Mark every feature this tree references as used in @p featureMap.
  void markFeaturesInMap(cv::Mat& featureMap);

 protected:
  /// Override that updates the cached feature value matrix when routing
  /// samples to children.
  void split_node_data(CvDTreeNode* n) override;
};
