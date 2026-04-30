/**
 * @file o_cvdtreeparams.h
 * @brief Parameters governing decision-tree training.
 */
#pragma once

/**
 * @brief Hyperparameters consumed by @ref CvDTree::train.
 *
 * Field summary:
 *  - @c max_categories: maximum cardinality of categorical splits before
 *    the trainer falls back to clustering.
 *  - @c max_depth: hard cap on tree depth (1 produces decision stumps,
 *    which is what cascade boosting normally uses).
 *  - @c min_sample_count: leaves with fewer samples than this are not split.
 *  - @c cv_folds: number of cross-validation folds used during cost-
 *    complexity pruning; @c 0 disables pruning.
 *  - @c use_surrogates: if @c true, learn surrogate splits to handle
 *    missing values at predict time.
 *  - @c use_1se_rule: pick the simplest tree within one standard error of
 *    the cv-optimal one (Breiman's 1-SE rule).
 *  - @c truncate_pruned_tree: physically discard pruned subtrees instead
 *    of keeping them as inactive nodes.
 *  - @c regression_accuracy: leaf-variance threshold below which a
 *    regression node is no longer split.
 *  - @c priors: per-class loss weights (length = number of classes); a
 *    null pointer means uniform priors.
 */
struct CvDTreeParams {
  int max_categories;
  int max_depth;
  int min_sample_count;
  int cv_folds;
  bool use_surrogates;
  bool use_1se_rule;
  bool truncate_pruned_tree;
  float regression_accuracy;
  const float* priors; ///< Optional per-class loss weights (not owned).

  CvDTreeParams();
  CvDTreeParams(int max_depth, int min_sample_count, float regression_accuracy,
                bool use_surrogates, int max_categories, int cv_folds,
                bool use_1se_rule, bool truncate_pruned_tree,
                const float* priors);
};