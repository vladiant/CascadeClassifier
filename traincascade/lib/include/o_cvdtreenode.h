/**
 * @file o_cvdtreenode.h
 * @brief In-memory node of a decision tree.
 */
#pragma once

struct CvDTreeNode;
struct CvDTreeSplit;

/**
 * @brief A single decision-tree node (internal or leaf).
 *
 * Holds: the tree structure (parent / left / right pointers), the split
 * predicate that routes samples (@c split, null for leaves), the
 * prediction value or class index emitted at this node, sample
 * bookkeeping and per-node statistics used by cost-complexity and
 * cross-validation pruning.
 */
struct CvDTreeNode {
  int class_idx; ///< Class index emitted at a leaf (classification mode).
  int Tn;        ///< Pruning truncation index used by cost-complexity pruning.
  double value;  ///< Leaf prediction value (regression) or class label (classification).

  CvDTreeNode* parent;
  CvDTreeNode* left;
  CvDTreeNode* right;

  CvDTreeSplit* split; ///< Primary split predicate; null on leaves.

  int sample_count;
  int depth;
  int* num_valid;     ///< Per-variable count of non-missing samples reaching this node.
  int offset;
  int buf_idx;
  double maxlr;

  // global pruning data
  int complexity;     ///< Subtree size used for cost-complexity scoring.
  double alpha;       ///< Cost-complexity threshold at which this subtree is pruned.
  double node_risk, tree_risk, tree_error;

  // cross-validation pruning data
  int* cv_Tn;           ///< Per-fold pruning level.
  double* cv_node_risk; ///< Per-fold node risk estimate.
  double* cv_node_error;///< Per-fold prediction error.

  /// Number of samples reaching this node with a valid value for variable @p vi;
  /// returns @c sample_count if no missing-data tracking is allocated.
  int get_num_valid(int vi) const { return num_valid ? num_valid[vi] : sample_count; }
  void set_num_valid(int vi, int n) {
    if (num_valid) num_valid[vi] = n;
  }
};
