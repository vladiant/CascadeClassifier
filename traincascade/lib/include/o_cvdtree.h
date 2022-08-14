#pragma once

#include <opencv2/core/core_c.h>

#include <memory>
#include <opencv2/core/core.hpp>

#include "o_cvdtreeparams.h"
#include "o_cvstatmodel.h"

class BlockedRange;
struct CvDTreeParams;
struct CvDTreeTrainData;
struct CvDTreeNode;
struct CvDTreeSplit;

// CvBoostTree
class CvDTree : public CvStatModel {
 public:
  CvDTree();
  virtual ~CvDTree();

  virtual bool train(const CvMat* trainData, int tflag, const CvMat* responses,
                     const CvMat* varIdx = 0, const CvMat* sampleIdx = 0,
                     const CvMat* varType = 0, const CvMat* missingDataMask = 0,
                     CvDTreeParams params = CvDTreeParams());

  virtual bool train(CvDTreeTrainData* trainData, const CvMat* subsampleIdx);

  virtual CvDTreeNode* predict(const CvMat* sample,
                               const CvMat* missingDataMask = 0,
                               bool preprocessedInput = false) const;

  virtual bool train(const cv::Mat& trainData, int tflag,
                     const cv::Mat& responses,
                     const cv::Mat& varIdx = cv::Mat(),
                     const cv::Mat& sampleIdx = cv::Mat(),
                     const cv::Mat& varType = cv::Mat(),
                     const cv::Mat& missingDataMask = cv::Mat(),
                     CvDTreeParams params = CvDTreeParams());

  virtual CvDTreeNode* predict(const cv::Mat& sample,
                               const cv::Mat& missingDataMask = cv::Mat(),
                               bool preprocessedInput = false) const;

  void clear() override;

  const CvDTreeNode* get_root() const;
  CvDTreeTrainData* get_data();

 protected:
  struct DTreeBestSplitFinder {
    DTreeBestSplitFinder(CvDTree* _tree, CvDTreeNode* _node);
    void operator()(const BlockedRange& range);
    std::shared_ptr<CvDTreeSplit> bestSplit;
    std::shared_ptr<CvDTreeSplit> split;
    int splitSize;
    CvDTree* tree;
    CvDTreeNode* node;
  };

  friend struct DTreeBestSplitFinder;

  virtual bool do_train(const CvMat* _subsample_idx);

  virtual void try_split_node(CvDTreeNode* n);
  virtual void split_node_data(CvDTreeNode* n);
  virtual CvDTreeSplit* find_best_split(CvDTreeNode* n);
  virtual CvDTreeSplit* find_split_ord_class(CvDTreeNode* n, int vi,
                                             float init_quality = 0,
                                             CvDTreeSplit* _split = 0,
                                             uchar* ext_buf = 0);
  virtual CvDTreeSplit* find_split_cat_class(CvDTreeNode* n, int vi,
                                             float init_quality = 0,
                                             CvDTreeSplit* _split = 0,
                                             uchar* ext_buf = 0);
  virtual CvDTreeSplit* find_split_ord_reg(CvDTreeNode* n, int vi,
                                           float init_quality = 0,
                                           CvDTreeSplit* _split = 0,
                                           uchar* ext_buf = 0);
  virtual CvDTreeSplit* find_split_cat_reg(CvDTreeNode* n, int vi,
                                           float init_quality = 0,
                                           CvDTreeSplit* _split = 0,
                                           uchar* ext_buf = 0);
  virtual CvDTreeSplit* find_surrogate_split_ord(CvDTreeNode* n, int vi,
                                                 uchar* ext_buf = 0);
  virtual CvDTreeSplit* find_surrogate_split_cat(CvDTreeNode* n, int vi,
                                                 uchar* ext_buf = 0);
  virtual double calc_node_dir(CvDTreeNode* node);
  virtual void complete_node_dir(CvDTreeNode* node);
  virtual void cluster_categories(const int* vectors, int vector_count,
                                  int var_count, int* sums, int k,
                                  int* cluster_labels);

  virtual void calc_node_value(CvDTreeNode* node);

  virtual void prune_cv();
  virtual double update_tree_rnc(int T, int fold);
  virtual int cut_tree(int T, int fold, double min_alpha);
  virtual void free_prune_data(bool cut_tree);
  virtual void free_tree();

  CvDTreeNode* root;
  CvMat* var_importance;
  CvDTreeTrainData* data;
  CvMat train_data_hdr, responses_hdr;
  cv::Mat train_data_mat, responses_mat;

 public:
  int pruned_tree_idx;
};
