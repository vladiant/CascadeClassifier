/**
 * @file o_cvdtreesplit.h
 * @brief Split predicate stored on each internal decision-tree node.
 */
#pragma once

/**
 * @brief One split predicate attached to a @ref CvDTreeNode.
 *
 * @c CvDTreeSplit overlays an ordered split (@c ord — feature value @c c
 * compared against the @c split_point quantile) and a categorical split
 * (@c subset — bitmask listing which categories go to the left child).
 * Multiple splits may form a linked list via @c next, e.g. surrogate
 * splits used for handling missing values.
 */
struct CvDTreeSplit {
  int var_idx;       ///< Index of the variable this split tests.
  int condensed_idx; ///< Index after var_idx remapping (set internally).
  int inversed;      ///< When non-zero the left/right semantic is inverted.
  float quality;     ///< Split-quality score (Gini drop or variance drop).
  CvDTreeSplit* next; ///< Next surrogate split for the same node, or null.
  /// Threshold-based split parameters for ordered features.
  struct Ord {
      float c;          ///< Threshold value: samples with @c x[var_idx] <= c go left.
      int split_point;  ///< Sorted-sample index at which the threshold sits.
  };
  union {
    int subset[2]; ///< 64-bit bitmask listing left-going categories.
    Ord ord;
  };
};