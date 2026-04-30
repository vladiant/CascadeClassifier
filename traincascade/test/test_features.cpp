#include <doctest/doctest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

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

// ---------------------------------------------------------------------------
// setImage / operator() — numerical tests on synthetic images
//
// These tests exercise the feature-evaluation code path end-to-end:
//   1. evaluator.init(...)         — generates feature descriptors
//   2. evaluator.setImage(img, ..) — computes integral images / histograms
//   3. evaluator(featureIdx, idx)  — evaluates a feature at a sample
//
// Each evaluator has a property that holds for any uniform (constant)
// image, which lets us assert exact numerical values without depending on
// which feature index corresponds to which geometric layout.
// ---------------------------------------------------------------------------

TEST_CASE("CvHaarEvaluator::operator(): returns 0 for every feature on a constant image") {
  // Arrange: a constant image has zero variance, so calcNormFactor() is 0
  // and CvHaarEvaluator::operator() short-circuits to 0.0f.
  CvHaarFeatureParams params(CvHaarFeatureParams::BASIC);
  params.maxCatCount = 0;
  params.featSize = 1;
  CvHaarEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(24, 24));
  cv::Mat constImg(24, 24, CV_8UC1, cv::Scalar(128));

  // Act
  evaluator.setImage(constImg, /*clsLabel=*/1, /*idx=*/0);

  // Assert: every Haar feature evaluates to exactly 0 on a flat patch.
  bool allZero = true;
  for (int fi = 0; fi < evaluator.getNumFeatures(); ++fi) {
    if (evaluator(fi, 0) != 0.0f) {
      allZero = false;
      break;
    }
  }
  CHECK(allZero);
  CHECK(evaluator.getNumFeatures() > 0);
}

TEST_CASE("CvHaarEvaluator::operator(): returns at least one non-zero value on a textured image") {
  // Arrange: a vertical step edge has non-zero variance and breaks the
  // Haar feature symmetry — at least one feature must produce a non-zero
  // response, otherwise something is wrong with setImage / operator().
  CvHaarFeatureParams params(CvHaarFeatureParams::BASIC);
  CvHaarEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(24, 24));
  cv::Mat img(24, 24, CV_8UC1, cv::Scalar(0));
  img(cv::Rect(12, 0, 12, 24)).setTo(cv::Scalar(255));  // vertical step edge

  // Act
  evaluator.setImage(img, /*clsLabel=*/1, /*idx=*/0);

  // Assert
  bool foundNonZero = false;
  for (int fi = 0; fi < evaluator.getNumFeatures() && !foundNonZero; ++fi) {
    if (evaluator(fi, 0) != 0.0f) {
      foundNonZero = true;
    }
  }
  CHECK(foundNonZero);
}

TEST_CASE("CvHaarEvaluator::setImage: ALL mode also computes the tilted integral") {
  // Arrange: ALL mode adds tilted features; the evaluator must still return
  // 0 on a constant image because the tilted integral is also flat.
  CvHaarFeatureParams params(CvHaarFeatureParams::ALL);
  CvHaarEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(24, 24));
  cv::Mat constImg(24, 24, CV_8UC1, cv::Scalar(64));

  // Act
  evaluator.setImage(constImg, /*clsLabel=*/0, /*idx=*/0);

  // Assert: pick a couple of feature indices spanning the full range.
  REQUIRE(evaluator.getNumFeatures() > 1);
  CHECK(evaluator(0, 0) == doctest::Approx(0.0f));
  CHECK(evaluator(evaluator.getNumFeatures() - 1, 0) == doctest::Approx(0.0f));
  // And the class label was stored.
  CHECK(evaluator.getCls(0) == doctest::Approx(0.0f));
}

TEST_CASE("CvLBPEvaluator::operator(): returns 255 for every feature on a constant image") {
  // Arrange: on a uniform image every 3x3 block sum equals cval, so every
  // one of the 8 LBP comparisons (`>= cval`) is true. Result: 0xFF == 255.
  CvLBPFeatureParams params;
  CvLBPEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(24, 24));
  cv::Mat constImg(24, 24, CV_8UC1, cv::Scalar(50));

  // Act
  evaluator.setImage(constImg, /*clsLabel=*/1, /*idx=*/0);

  // Assert
  REQUIRE(evaluator.getNumFeatures() > 0);
  bool allMax = true;
  for (int fi = 0; fi < evaluator.getNumFeatures(); ++fi) {
    if (evaluator(fi, 0) != 255.0f) {
      allMax = false;
      break;
    }
  }
  CHECK(allMax);
}

TEST_CASE("CvLBPEvaluator::operator(): produces values < 255 on a non-constant image") {
  // Arrange: a horizontal step edge breaks the >= cval invariant for at
  // least one comparison in many features.
  CvLBPFeatureParams params;
  CvLBPEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(24, 24));
  cv::Mat img(24, 24, CV_8UC1, cv::Scalar(0));
  img(cv::Rect(0, 12, 24, 12)).setTo(cv::Scalar(200));

  // Act
  evaluator.setImage(img, /*clsLabel=*/1, /*idx=*/0);

  // Assert: at least one feature must encode a bit pattern other than 0xFF.
  bool foundNonMax = false;
  for (int fi = 0; fi < evaluator.getNumFeatures() && !foundNonMax; ++fi) {
    if (evaluator(fi, 0) < 255.0f) {
      foundNonMax = true;
    }
  }
  CHECK(foundNonMax);
}

TEST_CASE("CvLBPEvaluator: setImage isolates samples by index") {
  // Arrange: write two different images at indices 0 and 1, then verify
  // each sample's evaluation reflects the image stored at that index.
  CvLBPFeatureParams params;
  CvLBPEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/2, cv::Size(24, 24));
  cv::Mat constImg(24, 24, CV_8UC1, cv::Scalar(80));
  cv::Mat textImg(24, 24, CV_8UC1, cv::Scalar(0));
  textImg(cv::Rect(0, 12, 24, 12)).setTo(cv::Scalar(200));

  // Act
  evaluator.setImage(constImg, /*clsLabel=*/0, /*idx=*/0);
  evaluator.setImage(textImg, /*clsLabel=*/1, /*idx=*/1);

  // Assert: sample 0 (constant) -> all features == 255; sample 1 (textured)
  // -> at least one feature differs from sample 0.
  REQUIRE(evaluator.getNumFeatures() > 0);
  CHECK(evaluator(0, 0) == doctest::Approx(255.0f));
  bool sample1HasDifferentValue = false;
  for (int fi = 0; fi < evaluator.getNumFeatures(); ++fi) {
    if (evaluator(fi, 1) != evaluator(fi, 0)) {
      sample1HasDifferentValue = true;
      break;
    }
  }
  CHECK(sample1HasDifferentValue);
  CHECK(evaluator.getCls(0) == doctest::Approx(0.0f));
  CHECK(evaluator.getCls(1) == doctest::Approx(1.0f));
}

TEST_CASE("CvHOGEvaluator::operator(): returns 0 for every component on a constant image") {
  // Arrange: a constant image has zero gradients, so every HOG bin is 0
  // and the implementation's `res > 0.001f` guard returns 0.0f.
  CvHOGFeatureParams params;
  CvHOGEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(32, 32));
  cv::Mat constImg(32, 32, CV_8UC1, cv::Scalar(100));

  // Act
  evaluator.setImage(constImg, /*clsLabel=*/1, /*idx=*/0);

  // Assert: getNumFeatures() returns the number of feature blocks, while
  // operator() is indexed by varIdx in [0, numFeatures * N_BINS * N_CELLS).
  REQUIRE(evaluator.getNumFeatures() > 0);
  const int totalVars = evaluator.getNumFeatures() * N_BINS * N_CELLS;
  bool allZero = true;
  for (int v = 0; v < totalVars; ++v) {
    if (evaluator(v, 0) != 0.0f) {
      allZero = false;
      break;
    }
  }
  CHECK(allZero);
}

TEST_CASE("CvHOGEvaluator::operator(): produces at least one non-zero on a textured image") {
  // Arrange
  CvHOGFeatureParams params;
  CvHOGEvaluator evaluator;
  evaluator.init(&params, /*maxSampleCount=*/1, cv::Size(32, 32));
  cv::Mat img(32, 32, CV_8UC1, cv::Scalar(0));
  img(cv::Rect(16, 0, 16, 32)).setTo(cv::Scalar(255));  // strong vertical edge

  // Act
  evaluator.setImage(img, /*clsLabel=*/1, /*idx=*/0);

  // Assert
  REQUIRE(evaluator.getNumFeatures() > 0);
  const int totalVars = evaluator.getNumFeatures() * N_BINS * N_CELLS;
  bool foundNonZero = false;
  for (int v = 0; v < totalVars && !foundNonZero; ++v) {
    if (evaluator(v, 0) > 0.0f) {
      foundNonZero = true;
    }
  }
  CHECK(foundNonZero);
}


// ---------------------------------------------------------------------------
// Direct CvHaarEvaluator::Feature::calc tests against known integral images
//
// Feature is a protected nested type, so we expose it via a thin probe
// subclass and construct/evaluate features by hand. The integral image is
// passed as a single flattened row (row-major) because calc() expects all
// fast-rect offsets to index into one cv::Mat row.
// ---------------------------------------------------------------------------

namespace {

class HaarFeatureProbe : public CvHaarEvaluator {
 public:
  using CvHaarEvaluator::Feature;
};
using HaarFeature = HaarFeatureProbe::Feature;

}  // namespace

TEST_CASE("CvHaarEvaluator::Feature::calc: upright two-rect feature on a vertical-step image") {
  // Arrange: 8x8 image, left half = 0, right half = 100. Feature: +1 over the
  // left half rectangle, -1 over the right half. Expected response =
  // (left sum) - (right sum) = 0 - (100 * 4 * 8) = -3200.
  cv::Mat img(8, 8, CV_8UC1, cv::Scalar(0));
  img.colRange(4, 8).setTo(100);
  cv::Mat sum;
  cv::integral(img, sum, CV_32S);                    // 9x9 CV_32S
  const cv::Mat sumRow = sum.reshape(0, 1);          // flatten to one row
  cv::Mat unusedTilted;                              // not read for upright
  const int offset = sum.cols;                       // = 9

  HaarFeature feature(offset, /*tilted=*/false,
                      /*x0,y0,w0,h0,wt0=*/0, 0, 4, 8, +1.0F,
                      /*x1,y1,w1,h1,wt1=*/4, 0, 4, 8, -1.0F);

  // Act
  const float response = feature.calc(sumRow, unusedTilted, 0);

  // Assert
  CHECK(response == doctest::Approx(-3200.0F));
}

TEST_CASE("CvHaarEvaluator::Feature::calc: upright feature returns zero on a uniform image") {
  // Arrange: uniform 8x8 image, balanced two-rect feature → response = 0.
  cv::Mat img(8, 8, CV_8UC1, cv::Scalar(42));
  cv::Mat sum;
  cv::integral(img, sum, CV_32S);
  const cv::Mat sumRow = sum.reshape(0, 1);
  cv::Mat unusedTilted;

  HaarFeature feature(sum.cols, /*tilted=*/false,
                      0, 0, 4, 8, +1.0F,
                      4, 0, 4, 8, -1.0F);

  // Act
  const float response = feature.calc(sumRow, unusedTilted, 0);

  // Assert: any balanced two-rect filter is zero on a constant image.
  CHECK(response == doctest::Approx(0.0F));
}

TEST_CASE("CvHaarEvaluator::Feature::calc: upright three-rect feature uses rect[2] when its weight is non-zero") {
  // Arrange: 9x3 image with the centre column = 200, others = 0. Build a
  // horizontal three-rect feature
  //   rect[0] = full 9x3   weight = +1
  //   rect[1] = centre 3x3 weight = -3
  // (centred-band Haar feature). On this 3x9 image:
  //   rect[0] sum = 200 * 3 (cols) * 3 (rows) = 1800
  //   rect[1] sum = 200 * 3 (cols) * 3 (rows) = 1800
  //   response   = 1800*1 + 1800*(-3) = -3600.
  cv::Mat img(3, 9, CV_8UC1, cv::Scalar(0));
  img.colRange(3, 6).setTo(200);
  cv::Mat sum;
  cv::integral(img, sum, CV_32S);                    // 4x10 CV_32S
  const cv::Mat sumRow = sum.reshape(0, 1);
  cv::Mat unusedTilted;

  HaarFeature feature(sum.cols, /*tilted=*/false,
                      /*rect0=*/0, 0, 9, 3, +1.0F,
                      /*rect1=*/3, 0, 3, 3, -3.0F);

  // Act
  const float response = feature.calc(sumRow, unusedTilted, 0);

  // Assert
  CHECK(response == doctest::Approx(-3600.0F));
}

TEST_CASE("CvHaarEvaluator::Feature::calc: tilted-feature branch reads the tilted integral image") {
  // Arrange: 16x16 uniform image of ones. The tilted integral image computed
  // by cv::integral lets us evaluate a 45-degree rotated rectangle's area as
  //     tilted[p0] + tilted[p3] - tilted[p1] - tilted[p2]
  // which on a unit-valued image equals w * h. We pick a rectangle that fits
  // entirely inside the image and use a single weighted rect (rect[1] has
  // zero weight, so its contribution drops out).
  cv::Mat img(16, 16, CV_8UC1, cv::Scalar(1));
  cv::Mat sum;
  cv::Mat sqsum;
  cv::Mat tilted;
  cv::integral(img, sum, sqsum, tilted, CV_32S);     // tilted: 17x17 CV_32S
  const cv::Mat tiltedRow = tilted.reshape(0, 1);
  cv::Mat unusedSum;                                 // not read for tilted

  // Tilted rectangle anchored so that all four corner offsets fall within the
  // 17x17 tilted integral. With x=8,y=2,w=4,h=4 the corners land at
  //   (8,2)  (12,6)  (4,6)  (8,10)  — all inside.
  HaarFeature feature(tilted.cols, /*tilted=*/true,
                      /*rect0=*/8, 2, 4, 4, +1.0F,
                      /*rect1 weight = 0 → ignored*/0, 0, 0, 0, 0.0F);

  // Act
  const float response = feature.calc(unusedSum, tiltedRow, 0);

  // Assert: the cascade-trainer tilted-rect convention has sides of length
  // w*sqrt(2) and h*sqrt(2) (w runs along (+1,+1), h along (-1,+1)), so the
  // rotated rectangle's area on an all-ones image is 2 * w * h = 32.
  CHECK(response == doctest::Approx(32.0F));
}
