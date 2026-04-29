#include <doctest/doctest.h>

#include <climits>

#include <opencv2/ml/ml.hpp>

#include "o_cvdtreeparams.h"
#include "o_cvboostparams.h"
#include "o_cvboost.h"
#include "boost.h"
#include "cascadeclassifier.h"

// ---------------------------------------------------------------------------
// CvDTreeParams
// ---------------------------------------------------------------------------

TEST_CASE("CvDTreeParams: default constructor sets documented defaults") {
  // Arrange / Act
  CvDTreeParams p;

  // Assert
  CHECK(p.max_categories == 10);
  CHECK(p.max_depth == INT_MAX);
  CHECK(p.min_sample_count == 10);
  CHECK(p.cv_folds == 10);
  CHECK(p.use_surrogates == true);
  CHECK(p.use_1se_rule == true);
  CHECK(p.truncate_pruned_tree == true);
  CHECK(p.regression_accuracy == doctest::Approx(0.01f));
  CHECK(p.priors == nullptr);
}

TEST_CASE("CvDTreeParams: parameterized constructor stores all arguments") {
  // Arrange
  const float priors[] = {0.5f, 0.5f};

  // Act
  CvDTreeParams p(/*max_depth=*/5,
                  /*min_sample_count=*/20,
                  /*regression_accuracy=*/0.05f,
                  /*use_surrogates=*/false,
                  /*max_categories=*/15,
                  /*cv_folds=*/3,
                  /*use_1se_rule=*/false,
                  /*truncate_pruned_tree=*/false,
                  priors);

  // Assert
  CHECK(p.max_depth == 5);
  CHECK(p.min_sample_count == 20);
  CHECK(p.regression_accuracy == doctest::Approx(0.05f));
  CHECK(p.use_surrogates == false);
  CHECK(p.max_categories == 15);
  CHECK(p.cv_folds == 3);
  CHECK(p.use_1se_rule == false);
  CHECK(p.truncate_pruned_tree == false);
  CHECK(p.priors == priors);
}

// ---------------------------------------------------------------------------
// CvBoostParams
// ---------------------------------------------------------------------------

TEST_CASE("CvBoostParams: default constructor selects REAL boost & one-level trees") {
  // Arrange / Act
  CvBoostParams p;

  // Assert
  CHECK(p.boost_type == cv::ml::Boost::REAL);
  CHECK(p.weak_count == 100);
  CHECK(p.weight_trim_rate == doctest::Approx(0.95));
  CHECK(p.cv_folds == 0);
  CHECK(p.max_depth == 1);
  CHECK(p.split_criteria == CvBoost::DEFAULT);
}

TEST_CASE("CvBoostParams: parameterized constructor copies arguments") {
  // Arrange
  const float priors[] = {0.3f, 0.7f};

  // Act
  CvBoostParams p(cv::ml::Boost::GENTLE,
                  /*weak_count=*/42,
                  /*weight_trim_rate=*/0.5,
                  /*max_depth=*/3,
                  /*use_surrogates=*/true,
                  priors);

  // Assert
  CHECK(p.boost_type == cv::ml::Boost::GENTLE);
  CHECK(p.weak_count == 42);
  CHECK(p.weight_trim_rate == doctest::Approx(0.5));
  CHECK(p.max_depth == 3);
  CHECK(p.use_surrogates == true);
  CHECK(p.priors == priors);
  CHECK(p.split_criteria == CvBoost::DEFAULT);
  CHECK(p.cv_folds == 0);
}

// ---------------------------------------------------------------------------
// CvCascadeBoostParams
// ---------------------------------------------------------------------------

TEST_CASE("CvCascadeBoostParams: default constructor uses GENTLE boost & cascade defaults") {
  // Arrange / Act
  CvCascadeBoostParams p;

  // Assert
  CHECK(p.boost_type == cv::ml::Boost::GENTLE);
  CHECK(p.minHitRate == doctest::Approx(0.995f));
  CHECK(p.maxFalseAlarm == doctest::Approx(0.5f));
  CHECK(p.use_surrogates == false);
  CHECK(p.use_1se_rule == false);
  CHECK(p.truncate_pruned_tree == false);
}

TEST_CASE("CvCascadeBoostParams: parameterized constructor stores cascade-specific values") {
  // Arrange / Act
  CvCascadeBoostParams p(cv::ml::Boost::DISCRETE,
                         /*minHitRate=*/0.99f,
                         /*maxFalseAlarm=*/0.4f,
                         /*weightTrimRate=*/0.95,
                         /*maxDepth=*/2,
                         /*maxWeakCount=*/50);

  // Assert: boost_type is forced to GENTLE in the cascade ctor regardless of input.
  CHECK(p.boost_type == cv::ml::Boost::GENTLE);
  CHECK(p.minHitRate == doctest::Approx(0.99f));
  CHECK(p.maxFalseAlarm == doctest::Approx(0.4f));
  CHECK(p.weight_trim_rate == doctest::Approx(0.95));
  CHECK(p.max_depth == 2);
  CHECK(p.weak_count == 50);
  CHECK(p.use_surrogates == false);
}

TEST_CASE("CvCascadeBoostParams::scanAttr: parses each named attribute") {
  // Arrange
  CvCascadeBoostParams p;

  // Act / Assert
  CHECK(p.scanAttr("-bt", "DAB"));
  CHECK(p.boost_type == cv::ml::Boost::DISCRETE);

  CHECK(p.scanAttr("-bt", "RAB"));
  CHECK(p.boost_type == cv::ml::Boost::REAL);

  CHECK(p.scanAttr("-bt", "LB"));
  CHECK(p.boost_type == cv::ml::Boost::LOGIT);

  CHECK(p.scanAttr("-bt", "GAB"));
  CHECK(p.boost_type == cv::ml::Boost::GENTLE);

  CHECK(p.scanAttr("-minHitRate", "0.9"));
  CHECK(p.minHitRate == doctest::Approx(0.9f));

  CHECK(p.scanAttr("-maxFalseAlarmRate", "0.25"));
  CHECK(p.maxFalseAlarm == doctest::Approx(0.25f));

  CHECK(p.scanAttr("-weightTrimRate", "0.8"));
  CHECK(p.weight_trim_rate == doctest::Approx(0.8));

  CHECK(p.scanAttr("-maxDepth", "7"));
  CHECK(p.max_depth == 7);

  CHECK(p.scanAttr("-maxWeakCount", "12"));
  CHECK(p.weak_count == 12);
}

TEST_CASE("CvCascadeBoostParams::scanAttr: rejects unknown args and bad boost type") {
  // Arrange
  CvCascadeBoostParams p;

  // Act / Assert
  CHECK_FALSE(p.scanAttr("-unknown", "value"));
  CHECK_FALSE(p.scanAttr("-bt", "NOPE"));
}

// ---------------------------------------------------------------------------
// CvCascadeParams
// ---------------------------------------------------------------------------

TEST_CASE("CvCascadeParams: default constructor uses BOOST stages, HAAR features, 24x24") {
  // Arrange / Act
  CvCascadeParams p;

  // Assert
  CHECK(p.stageType == CvCascadeParams::BOOST);
  CHECK(p.featureType == CvFeatureParams::HAAR);
  CHECK(p.winSize.width == 24);
  CHECK(p.winSize.height == 24);
}

TEST_CASE("CvCascadeParams: parameterized constructor stores stage and feature types") {
  // Arrange / Act
  CvCascadeParams p(CvCascadeParams::BOOST, CvFeatureParams::LBP);

  // Assert
  CHECK(p.stageType == CvCascadeParams::BOOST);
  CHECK(p.featureType == CvFeatureParams::LBP);
  CHECK(p.winSize.width == 24);
  CHECK(p.winSize.height == 24);
}

TEST_CASE("CvCascadeParams::scanAttr: parses width, height and feature type") {
  // Arrange
  CvCascadeParams p;

  // Act / Assert
  CHECK(p.scanAttr("-w", "32"));
  CHECK(p.winSize.width == 32);

  CHECK(p.scanAttr("-h", "48"));
  CHECK(p.winSize.height == 48);

  CHECK(p.scanAttr("-featureType", "LBP"));
  CHECK(p.featureType == CvFeatureParams::LBP);

  CHECK(p.scanAttr("-featureType", "HOG"));
  CHECK(p.featureType == CvFeatureParams::HOG);

  CHECK(p.scanAttr("-stageType", "BOOST"));
  CHECK(p.stageType == CvCascadeParams::BOOST);
}

TEST_CASE("CvCascadeParams::scanAttr: returns false for unknown attribute") {
  // Arrange
  CvCascadeParams p;

  // Act
  const bool result = p.scanAttr("-bogus", "0");

  // Assert
  CHECK_FALSE(result);
}
