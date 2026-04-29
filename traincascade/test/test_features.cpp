#include <doctest/doctest.h>

#include <opencv2/core.hpp>

#include "traincascade_features.h"
#include "haarfeatures.h"
#include "lbpfeatures.h"
#include "HOGfeatures.h"

// ---------------------------------------------------------------------------
// CvFeatureParams factory + base behavior
// ---------------------------------------------------------------------------

TEST_CASE("CvFeatureParams::create: returns a HAAR-typed params object") {
  // Arrange / Act
  cv::Ptr<CvFeatureParams> p = CvFeatureParams::create(CvFeatureParams::HAAR);

  // Assert
  REQUIRE(p);
  // featSize for simple features is 1.
  CHECK(p->featSize == 1);
  CHECK(dynamic_cast<CvHaarFeatureParams*>(p.get()) != nullptr);
}

TEST_CASE("CvFeatureParams::create: returns an LBP-typed params object with maxCatCount=256") {
  // Arrange / Act
  cv::Ptr<CvFeatureParams> p = CvFeatureParams::create(CvFeatureParams::LBP);

  // Assert
  REQUIRE(p);
  CHECK(p->maxCatCount == 256);
  CHECK(dynamic_cast<CvLBPFeatureParams*>(p.get()) != nullptr);
}

TEST_CASE("CvFeatureParams::create: returns a HOG-typed params object with featSize=N_BINS*N_CELLS") {
  // Arrange / Act
  cv::Ptr<CvFeatureParams> p = CvFeatureParams::create(CvFeatureParams::HOG);

  // Assert
  REQUIRE(p);
  CHECK(p->maxCatCount == 0);
  CHECK(p->featSize == N_BINS * N_CELLS);
  CHECK(dynamic_cast<CvHOGFeatureParams*>(p.get()) != nullptr);
}

TEST_CASE("CvFeatureParams::create: returns an empty Ptr for an unknown feature type") {
  // Arrange / Act
  cv::Ptr<CvFeatureParams> p = CvFeatureParams::create(99);

  // Assert
  CHECK(!p);
}

TEST_CASE("CvFeatureEvaluator::create: returns concrete evaluators for known feature types") {
  // Arrange / Act
  cv::Ptr<CvFeatureEvaluator> haar =
      CvFeatureEvaluator::create(CvFeatureParams::HAAR);
  cv::Ptr<CvFeatureEvaluator> lbp =
      CvFeatureEvaluator::create(CvFeatureParams::LBP);
  cv::Ptr<CvFeatureEvaluator> hog =
      CvFeatureEvaluator::create(CvFeatureParams::HOG);
  cv::Ptr<CvFeatureEvaluator> bad = CvFeatureEvaluator::create(123);

  // Assert
  CHECK(haar);
  CHECK(lbp);
  CHECK(hog);
  CHECK(!bad);
}

// ---------------------------------------------------------------------------
// CvHaarFeatureParams
// ---------------------------------------------------------------------------

TEST_CASE("CvHaarFeatureParams: default constructor selects BASIC mode") {
  // Arrange / Act
  CvHaarFeatureParams p;

  // Assert
  CHECK(p.mode == CvHaarFeatureParams::BASIC);
}

TEST_CASE("CvHaarFeatureParams: explicit constructor stores requested mode") {
  // Arrange / Act
  CvHaarFeatureParams basic(CvHaarFeatureParams::BASIC);
  CvHaarFeatureParams core(CvHaarFeatureParams::CORE);
  CvHaarFeatureParams all(CvHaarFeatureParams::ALL);

  // Assert
  CHECK(basic.mode == CvHaarFeatureParams::BASIC);
  CHECK(core.mode == CvHaarFeatureParams::CORE);
  CHECK(all.mode == CvHaarFeatureParams::ALL);
}

TEST_CASE("CvHaarFeatureParams::scanAttr: rejects an unknown mode value") {
  // Arrange
  CvHaarFeatureParams p;

  // Act
  const bool result = p.scanAttr("-mode", "GARBAGE");

  // Assert
  CHECK_FALSE(result);
  CHECK(p.mode == -1);
}

TEST_CASE("CvHaarFeatureParams::init: copies fields from another instance") {
  // Arrange
  CvHaarFeatureParams src(CvHaarFeatureParams::ALL);
  src.maxCatCount = 7;
  src.featSize = 3;
  CvHaarFeatureParams dst;

  // Act
  dst.init(src);

  // Assert
  CHECK(dst.mode == CvHaarFeatureParams::ALL);
  CHECK(dst.maxCatCount == 7);
  CHECK(dst.featSize == 3);
}

// ---------------------------------------------------------------------------
// CvLBPFeatureParams / CvHOGFeatureParams
// ---------------------------------------------------------------------------

TEST_CASE("CvLBPFeatureParams: default constructor sets maxCatCount=256") {
  // Arrange / Act
  CvLBPFeatureParams p;

  // Assert
  CHECK(p.maxCatCount == 256);
  CHECK(p.featSize == 1);
}

TEST_CASE("CvHOGFeatureParams: default constructor sets featSize=N_BINS*N_CELLS") {
  // Arrange / Act
  CvHOGFeatureParams p;

  // Assert
  CHECK(p.maxCatCount == 0);
  CHECK(p.featSize == N_BINS * N_CELLS);
}

// ---------------------------------------------------------------------------
// Evaluator init() and generateFeatures()
// ---------------------------------------------------------------------------

TEST_CASE("CvHaarEvaluator::init: generates a non-empty feature set for a 24x24 window") {
  // Arrange
  CvHaarFeatureParams params(CvHaarFeatureParams::BASIC);
  params.maxCatCount = 0;
  params.featSize = 1;
  CvHaarEvaluator evaluator;

  // Act
  evaluator.init(&params, /*maxSampleCount=*/4, cv::Size(24, 24));

  // Assert
  CHECK(evaluator.getNumFeatures() > 0);
  CHECK(evaluator.getFeatureSize() == 1);
  CHECK(evaluator.getMaxCatCount() == 0);
  CHECK(evaluator.getCls().rows == 4);
  CHECK(evaluator.getCls().cols == 1);
}

TEST_CASE("CvHaarEvaluator: BASIC and ALL produce different feature counts") {
  // Arrange
  CvHaarFeatureParams basic(CvHaarFeatureParams::BASIC);
  basic.maxCatCount = 0;
  basic.featSize = 1;
  CvHaarFeatureParams all(CvHaarFeatureParams::ALL);
  all.maxCatCount = 0;
  all.featSize = 1;
  CvHaarEvaluator basicEval;
  CvHaarEvaluator allEval;

  // Act
  basicEval.init(&basic, 1, cv::Size(24, 24));
  allEval.init(&all, 1, cv::Size(24, 24));

  // Assert: ALL mode includes tilted features and is strictly larger.
  CHECK(allEval.getNumFeatures() > basicEval.getNumFeatures());
}

TEST_CASE("CvLBPEvaluator::init: generates a non-empty feature set for a 24x24 window") {
  // Arrange
  CvLBPFeatureParams params;
  CvLBPEvaluator evaluator;

  // Act
  evaluator.init(&params, /*maxSampleCount=*/2, cv::Size(24, 24));

  // Assert
  CHECK(evaluator.getNumFeatures() > 0);
  CHECK(evaluator.getMaxCatCount() == 256);
}

TEST_CASE("CvHOGEvaluator::init: generates a non-empty feature set for a 32x32 window") {
  // Arrange
  CvHOGFeatureParams params;
  CvHOGEvaluator evaluator;

  // Act
  evaluator.init(&params, /*maxSampleCount=*/2, cv::Size(32, 32));

  // Assert: HOG features only generated when winSize/2 >= 8.
  CHECK(evaluator.getNumFeatures() > 0);
  CHECK(evaluator.getFeatureSize() == N_BINS * N_CELLS);
}

TEST_CASE("CvHOGEvaluator::init: generates no features when window is too small") {
  // Arrange: HOG inner loop requires winSize.width/2 >= 8.
  CvHOGFeatureParams params;
  CvHOGEvaluator evaluator;

  // Act
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(8, 8));

  // Assert
  CHECK(evaluator.getNumFeatures() == 0);
}

TEST_CASE("CvFeatureEvaluator::setImage: stores class label at the given sample index") {
  // Arrange
  CvLBPFeatureParams params;
  CvLBPEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/3, cv::Size(24, 24));
  cv::Mat img(24, 24, CV_8UC1, cv::Scalar(127));

  // Act
  evaluator.setImage(img, /*clsLabel=*/1, /*idx=*/2);

  // Assert
  CHECK(evaluator.getCls(2) == doctest::Approx(1.0f));
}
