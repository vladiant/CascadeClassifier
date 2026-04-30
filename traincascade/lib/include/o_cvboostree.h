/**
 * @file o_cvboostree.h
 * @brief Decision-tree weak learner used by @ref CvBoost / @ref CvCascadeBoost.
 */
#pragma once

#include "o_cvdtree.h"

class CvBoost;

/**
 * @brief Single decision tree trained as a weak learner inside a boosted
 *        ensemble.
 *
 * Inherits @ref CvDTree's split-finding machinery and overrides the
 * value-computation hooks (@c calc_node_value, @c calc_node_dir,
 * @c find_split_*) so the tree minimizes the weighted objective
 * implied by the parent ensemble's boosting variant. The standalone
 * @ref CvDTree::train overloads are kept around as no-ops to avoid
 * compiler warnings; only the @c train(trainData, idx, ensemble)
 * overload should be called from outside.
 */
class CvBoostTree : public CvDTree {
 public:
  CvBoostTree();
  virtual ~CvBoostTree();

  using CvDTree::train;
  /// Train this weak learner against the boosting state held by @p ensemble.
  bool train(CvDTreeTrainData* trainData, const CvMat* subsample_idx,
             CvBoost* ensemble);

  /// Multiply every leaf value by @p s (used by REAL/LOGIT AdaBoost).
  virtual void scale(double s);
  void clear() override;

  /* dummy methods to avoid warnings: BEGIN */
  bool train(const CvMat* trainData, int tflag, const CvMat* responses,
             const CvMat* varIdx = 0, const CvMat* sampleIdx = 0,
             const CvMat* varType = 0, const CvMat* missingDataMask = 0,
             CvDTreeParams params = CvDTreeParams()) override;
  bool train(CvDTreeTrainData* trainData, const CvMat* _subsample_idx) override;
  /* dummy methods to avoid warnings: END */

 protected:
  void try_split_node(CvDTreeNode* n) override;
  CvDTreeSplit* find_surrogate_split_ord(CvDTreeNode* n, int vi,
                                         uchar* ext_buf = 0) override;
  CvDTreeSplit* find_surrogate_split_cat(CvDTreeNode* n, int vi,
                                         uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_ord_class(CvDTreeNode* n, int vi,
                                     float init_quality = 0,
                                     CvDTreeSplit* _split = 0,
                                     uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_cat_class(CvDTreeNode* n, int vi,
                                     float init_quality = 0,
                                     CvDTreeSplit* _split = 0,
                                     uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_ord_reg(CvDTreeNode* n, int vi,
                                   float init_quality = 0,
                                   CvDTreeSplit* _split = 0,
                                   uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_cat_reg(CvDTreeNode* n, int vi,
                                   float init_quality = 0,
                                   CvDTreeSplit* _split = 0,
                                   uchar* ext_buf = 0) override;
  /// Compute the leaf prediction using the boosting weights.
  void calc_node_value(CvDTreeNode* n) override;
  /// Compute the routing direction (left/right majority) for node @p n.
  double calc_node_dir(CvDTreeNode* n) override;

  CvBoost* ensemble; ///< Non-owning back-pointer to the parent ensemble.
};
