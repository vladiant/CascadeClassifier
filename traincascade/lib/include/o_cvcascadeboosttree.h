#pragma once

#include "o_cvboostree.h"

struct CvDTreeNode;
class CvBoost;
class CvDTreeTrainData;

namespace cv {
class FileStorage;
class FileNode;
class Mat;
}  // namespace cv

// CvCascadeClassifier, CvCascadeBoost
class CvCascadeBoostTree : public CvBoostTree {
 public:
  virtual CvDTreeNode* predict(int sampleIdx) const;
  void write(cv::FileStorage& fs, const cv::Mat& featureMap);
  void read(const cv::FileNode& node, CvBoost* _ensemble,
            CvDTreeTrainData* _data);
  void markFeaturesInMap(cv::Mat& featureMap);

 protected:
  void split_node_data(CvDTreeNode* n) override;
};
