/**
 * @file o_cvboostparams.h
 * @brief Parameters of OpenCV's legacy boosted-tree ensemble (@c CvBoost).
 */
#pragma once

#include "o_cvdtreeparams.h"

/**
 * @brief Hyperparameters for boosted decision-tree training.
 *
 * Inherits the decision-tree settings from @ref CvDTreeParams (max
 * depth, min sample count per leaf, etc.) and adds:
 *  - @c boost_type: variant — DISCRETE, REAL, LOGIT or GENTLE AdaBoost.
 *  - @c weak_count: maximum number of weak learners in the ensemble.
 *  - @c split_criteria: per-split scoring (Gini, misclassification, SSE...).
 *  - @c weight_trim_rate: discard low-weight samples whose cumulative
 *    weight is below this rate to speed up training.
 */
struct CvBoostParams : public CvDTreeParams {
  int boost_type;          ///< AdaBoost variant (see @ref CvBoost).
  int weak_count;          ///< Maximum number of weak learners.
  int split_criteria;      ///< Split scoring criterion (Gini, misclass, SSE, ...).
  double weight_trim_rate; ///< Cumulative-weight trimming threshold.

  CvBoostParams();
  CvBoostParams(int boost_type, int weak_count, double weight_trim_rate,
                int max_depth, bool use_surrogates, const float* priors);
};
