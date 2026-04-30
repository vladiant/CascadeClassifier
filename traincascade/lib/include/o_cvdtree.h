/**
 * @file o_cvdtree.h
 * @brief Legacy OpenCV decision tree (@c CvDTree).
 *
 * Implements the standard CART training pipeline: best-split search,
 * recursive node splitting, optional surrogate splits for missing values,
 * leaf value computation, and post-training cost-complexity pruning with
 * cross-validation. Used directly inside the cascade as the base class
 * for @ref CvBoostTree (boosting weak learner) and indirectly via
 * @ref CvCascadeBoostTree.
 */
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

/**
 * @brief CART-style decision tree.
 *
 * The class supports both classification (categorical response) and
 * regression (ordered response) modes; the choice is made automatically
 * from the @p varType matrix passed to @ref train. Two training entry
 * points exist:
 *  - the high-level @ref train(const cv::Mat&, ...) overload, which
 *    builds a private @ref CvDTreeTrainData from raw matrices, and
 *  - the low-level @ref train(CvDTreeTrainData*, const CvMat*) overload,
 *    used by the boosting trainer to share a precomputed data store
 *    across all weak learners.
 */
class CvDTree : public CvStatModel {
 public:
  CvDTree();
  virtual ~CvDTree();

  /**
   * @brief CvMat-based training entry point (legacy API).
   *
   * @param trainData @c (sample_count x var_count) feature matrix.
   * @param tflag @c CV_ROW_SAMPLE or @c CV_COL_SAMPLE.
   * @param responses Response vector (categorical or ordered).
   * @param varIdx Optional subset of variables to use.
   * @param sampleIdx Optional subset of samples to use.
   * @param varType Per-variable type vector (last entry is the response type).
   * @param missingDataMask Optional 8U mask of missing values.
   * @param params Tree-training hyperparameters.
   * @return @c true on success.
   */
  virtual bool train(const CvMat* trainData, int tflag, const CvMat* responses,
                     const CvMat* varIdx = 0, const CvMat* sampleIdx = 0,
                     const CvMat* varType = 0, const CvMat* missingDataMask = 0,
                     CvDTreeParams params = CvDTreeParams());

  /// Train on an externally-built and possibly shared @ref CvDTreeTrainData.
  /// Used by the boosting trainer to amortize data preparation across weak learners.
  virtual bool train(CvDTreeTrainData* trainData, const CvMat* subsampleIdx);

  /**
   * @brief Predict the leaf reached by @p sample.
   * @param sample 1xvar_count or var_countx1 feature vector.
   * @param missingDataMask Optional mask (1 where a value is missing).
   * @param preprocessedInput If @c true the categorical fields of @p sample
   *        are already category indices (skip the lookup).
   * @return The leaf node reached or @c nullptr on failure.
   */
  virtual CvDTreeNode* predict(const CvMat* sample,
                               const CvMat* missingDataMask = 0,
                               bool preprocessedInput = false) const;

  /// @copydoc CvDTree::train(const CvMat*,int,const CvMat*,const CvMat*,const CvMat*,const CvMat*,const CvMat*,CvDTreeParams)
  virtual bool train(const cv::Mat& trainData, int tflag,
                     const cv::Mat& responses,
                     const cv::Mat& varIdx = cv::Mat(),
                     const cv::Mat& sampleIdx = cv::Mat(),
                     const cv::Mat& varType = cv::Mat(),
                     const cv::Mat& missingDataMask = cv::Mat(),
                     CvDTreeParams params = CvDTreeParams());

  /// @copydoc CvDTree::predict(const CvMat*,const CvMat*,bool) const
  virtual CvDTreeNode* predict(const cv::Mat& sample,
                               const cv::Mat& missingDataMask = cv::Mat(),
                               bool preprocessedInput = false) const;

  /// Release every internal buffer so the tree can be retrained.
  void clear() override;

  const CvDTreeNode* get_root() const;
  CvDTreeTrainData* get_data();

 protected:
  /**
   * @brief Functor that searches the best split among a range of variables.
   *
   * Used as the body of @ref parallel_reduce so the per-variable split
   * search can be parallelized when a real parallel runtime is plugged in.
   */
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

  /// Internal training driver shared by both public train overloads.
  virtual bool do_train(const CvMat* _subsample_idx);

  /// Recursive splitter: try to split @p n; if it qualifies, recurse on children.
  virtual void try_split_node(CvDTreeNode* n);
  /// Partition @p n's samples between its left/right children using @c n->split.
  virtual void split_node_data(CvDTreeNode* n);
  /// Search every variable for the best split of node @p n.
  virtual CvDTreeSplit* find_best_split(CvDTreeNode* n);
  /// Search the best split for an ordered variable in classification mode.
  virtual CvDTreeSplit* find_split_ord_class(CvDTreeNode* n, int vi,
                                             float init_quality = 0,
                                             CvDTreeSplit* _split = 0,
                                             uchar* ext_buf = 0);
  /// Search the best split for a categorical variable in classification mode.
  virtual CvDTreeSplit* find_split_cat_class(CvDTreeNode* n, int vi,
                                             float init_quality = 0,
                                             CvDTreeSplit* _split = 0,
                                             uchar* ext_buf = 0);
  /// Search the best split for an ordered variable in regression mode.
  virtual CvDTreeSplit* find_split_ord_reg(CvDTreeNode* n, int vi,
                                           float init_quality = 0,
                                           CvDTreeSplit* _split = 0,
                                           uchar* ext_buf = 0);
  /// Search the best split for a categorical variable in regression mode.
  virtual CvDTreeSplit* find_split_cat_reg(CvDTreeNode* n, int vi,
                                           float init_quality = 0,
                                           CvDTreeSplit* _split = 0,
                                           uchar* ext_buf = 0);
  /// Find an ordered-variable surrogate that approximates the primary split.
  virtual CvDTreeSplit* find_surrogate_split_ord(CvDTreeNode* n, int vi,
                                                 uchar* ext_buf = 0);
  /// Find a categorical-variable surrogate that approximates the primary split.
  virtual CvDTreeSplit* find_surrogate_split_cat(CvDTreeNode* n, int vi,
                                                 uchar* ext_buf = 0);
  /// Compute the routing direction (left/right majority) for samples reaching @p node.
  virtual double calc_node_dir(CvDTreeNode* node);
  /// Resolve the final left/right routing including surrogate-driven decisions.
  virtual void complete_node_dir(CvDTreeNode* node);
  /// k-means clustering of categories used for high-cardinality categorical splits.
  virtual void cluster_categories(const int* vectors, int vector_count,
                                  int var_count, int* sums, int k,
                                  int* cluster_labels);

  /// Compute the leaf prediction value (class label or regression mean) for @p node.
  virtual void calc_node_value(CvDTreeNode* node);

  /// Run cost-complexity pruning with cross-validation; populated by @c cv_folds.
  virtual void prune_cv();
  /// Update per-fold risk/node-count tables for pruning level @p T at fold @p fold.
  virtual double update_tree_rnc(int T, int fold);
  /// Cut the tree at every node whose alpha falls below @p min_alpha.
  virtual int cut_tree(int T, int fold, double min_alpha);
  /// Release pruning bookkeeping; if @p cut_tree is true also discard pruned subtrees.
  virtual void free_prune_data(bool cut_tree);
  /// Release the in-memory tree (root, nodes, splits).
  virtual void free_tree();

  CvDTreeNode* root{};
  CvMat* var_importance;
  CvDTreeTrainData* data;
  CvMat train_data_hdr{}, responses_hdr{};
  cv::Mat train_data_mat, responses_mat;

 public:
  int pruned_tree_idx{}; ///< Pruning truncation index in effect (0 = full tree).
};
