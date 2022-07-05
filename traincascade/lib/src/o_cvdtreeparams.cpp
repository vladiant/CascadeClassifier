#include "o_cvdtreeparams.h"

#include <climits>

CvDTreeParams::CvDTreeParams()
    : max_categories(10),
      max_depth(INT_MAX),
      min_sample_count(10),
      cv_folds(10),
      use_surrogates(true),
      use_1se_rule(true),
      truncate_pruned_tree(true),
      regression_accuracy(0.01f),
      priors(0) {}

CvDTreeParams::CvDTreeParams(int _max_depth, int _min_sample_count,
                             float _regression_accuracy, bool _use_surrogates,
                             int _max_categories, int _cv_folds,
                             bool _use_1se_rule, bool _truncate_pruned_tree,
                             const float* _priors)
    : max_categories(_max_categories),
      max_depth(_max_depth),
      min_sample_count(_min_sample_count),
      cv_folds(_cv_folds),
      use_surrogates(_use_surrogates),
      use_1se_rule(_use_1se_rule),
      truncate_pruned_tree(_truncate_pruned_tree),
      regression_accuracy(_regression_accuracy),
      priors(_priors) {}