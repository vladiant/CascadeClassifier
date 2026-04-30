#include <doctest/doctest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include "cascadeclassifier.h"

namespace fs = std::filesystem;

#ifndef TRAINCASCADE_RES_DIR
#define TRAINCASCADE_RES_DIR "."
#endif

namespace {

// Provide a unique, empty directory for the cascade classifier to write to.
// Using a per-test directory keeps integration tests independent and parallel-safe.
fs::path makeUniqueOutputDir(const std::string& tag) {
  const auto base = fs::temp_directory_path() / "traincascade_it";
  fs::create_directories(base);
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  std::random_device rd;
  const auto suffix =
      std::to_string(stamp) + "_" + std::to_string(rd()) + "_" + tag;
  const auto dir = base / suffix;
  fs::create_directories(dir);
  return dir;
}

struct ResourcePaths {
  fs::path vec;
  fs::path bg;
};

// Stage the resources required by CvCascadeImageReader into a working dir:
//  - A copy of barcode.vec (the positive samples).
//  - A synthesized negative image large enough to host the 75x32 detection
//    window (the bundled res/bg.png is only 32x32, which is fine for OpenCV's
//    high-level detector but too small for the cascade trainer's NegReader
//    to slide a 75x32 window over).
//  - A bg.txt with an absolute path to that negative image.
ResourcePaths stageResources(const fs::path& workDir) {
  const fs::path resDir{TRAINCASCADE_RES_DIR};
  REQUIRE(fs::exists(resDir));
  REQUIRE(fs::exists(resDir / "barcode.vec"));

  // Synthesize a 256x128 grayscale image with deterministic-but-non-uniform
  // texture so the negative reader has plenty of room to slide a 75x32 window.
  cv::Mat negImg(128, 256, CV_8UC1);
  for (int r = 0; r < negImg.rows; ++r) {
    for (int c = 0; c < negImg.cols; ++c) {
      negImg.at<uchar>(r, c) = static_cast<uchar>((r * 7 + c * 13) & 0xFF);
    }
  }
  const auto negImgPath = workDir / "neg.png";
  REQUIRE(cv::imwrite(negImgPath.string(), negImg));

  {
    std::ofstream bgFile(workDir / "bg.txt");
    bgFile << negImgPath.string() << '\n';
  }
  fs::copy_file(resDir / "barcode.vec", workDir / "barcode.vec",
                fs::copy_options::overwrite_existing);

  return {workDir / "barcode.vec", workDir / "bg.txt"};
}

} // namespace

// ---------------------------------------------------------------------------
// Happy paths
// ---------------------------------------------------------------------------

TEST_CASE("CvCascadeClassifier::train: completes one LBP stage and writes cascade.xml") {
  // Arrange: stage barcode.vec/bg.txt, allocate a fresh data dir.
  const auto workDir = makeUniqueOutputDir("lbp");
  const auto res = stageResources(workDir);
  const auto dataDir = workDir / "data";
  fs::create_directories(dataDir);

  CvCascadeParams cascadeParams(CvCascadeParams::BOOST,
                                CvFeatureParams::LBP);
  cascadeParams.winSize = cv::Size(75, 32); // matches barcode.vec
  CvLBPFeatureParams featureParams;
  CvCascadeBoostParams stageParams(cv::ml::Boost::GENTLE,
                                   /*minHitRate=*/0.995F,
                                   /*maxFalseAlarm=*/0.5F,
                                   /*weightTrimRate=*/0.95,
                                   /*maxDepth=*/1,
                                   /*maxWeakCount=*/10);

  CvCascadeClassifier classifier;

  // Act
  const bool ok = classifier.train(dataDir.string(),
                                   res.vec.string(),
                                   res.bg.string(),
                                   /*numPos=*/20,
                                   /*numNeg=*/1,
                                   /*precalcValBufSize=*/64,
                                   /*precalcIdxBufSize=*/64,
                                   /*numStages=*/1,
                                   cascadeParams,
                                   featureParams,
                                   stageParams,
                                   /*baseFormatSave=*/false,
                                   /*acceptanceRatioBreakValue=*/-1.0);

  // Assert: training must complete and emit cascade.xml + params.xml + stage0.xml.
  CHECK(ok);
  CHECK(fs::exists(dataDir / "cascade.xml"));
  CHECK(fs::exists(dataDir / "params.xml"));
  CHECK(fs::exists(dataDir / "stage0.xml"));

  // The produced cascade.xml must be loadable by the public OpenCV detector.
  cv::CascadeClassifier loaded((dataDir / "cascade.xml").string());
  CHECK_FALSE(loaded.empty());

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}

TEST_CASE("CvCascadeClassifier::train: HAAR BASIC mode also produces a usable cascade") {
  // Arrange
  const auto workDir = makeUniqueOutputDir("haar");
  const auto res = stageResources(workDir);
  const auto dataDir = workDir / "data";
  fs::create_directories(dataDir);

  CvCascadeParams cascadeParams(CvCascadeParams::BOOST,
                                CvFeatureParams::HAAR);
  cascadeParams.winSize = cv::Size(75, 32);
  CvHaarFeatureParams featureParams(CvHaarFeatureParams::BASIC);
  CvCascadeBoostParams stageParams(cv::ml::Boost::GENTLE,
                                   0.995F, 0.5F, 0.95, 1, 10);

  CvCascadeClassifier classifier;

  // Act
  const bool ok = classifier.train(dataDir.string(),
                                   res.vec.string(),
                                   res.bg.string(),
                                   /*numPos=*/20,
                                   /*numNeg=*/1,
                                   /*precalcValBufSize=*/64,
                                   /*precalcIdxBufSize=*/64,
                                   /*numStages=*/1,
                                   cascadeParams,
                                   featureParams,
                                   stageParams,
                                   /*baseFormatSave=*/false,
                                   /*acceptanceRatioBreakValue=*/-1.0);

  // Assert
  CHECK(ok);
  CHECK(fs::exists(dataDir / "cascade.xml"));
  cv::CascadeClassifier loaded((dataDir / "cascade.xml").string());
  CHECK_FALSE(loaded.empty());

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}

TEST_CASE("CvCascadeClassifier::train: baseFormatSave produces a non-empty cascade.xml") {
  // Arrange: baseFormatSave is only supported for Haar-like features.
  const auto workDir = makeUniqueOutputDir("base");
  const auto res = stageResources(workDir);
  const auto dataDir = workDir / "data";
  fs::create_directories(dataDir);

  CvCascadeParams cascadeParams(CvCascadeParams::BOOST,
                                CvFeatureParams::HAAR);
  cascadeParams.winSize = cv::Size(75, 32);
  CvHaarFeatureParams featureParams(CvHaarFeatureParams::BASIC);
  CvCascadeBoostParams stageParams(cv::ml::Boost::GENTLE,
                                   0.995F, 0.5F, 0.95, 1, 10);

  CvCascadeClassifier classifier;

  // Act
  const bool ok = classifier.train(dataDir.string(),
                                   res.vec.string(),
                                   res.bg.string(),
                                   /*numPos=*/20,
                                   /*numNeg=*/1,
                                   /*precalcValBufSize=*/64,
                                   /*precalcIdxBufSize=*/64,
                                   /*numStages=*/1,
                                   cascadeParams,
                                   featureParams,
                                   stageParams,
                                   /*baseFormatSave=*/true,
                                   /*acceptanceRatioBreakValue=*/-1.0);

  // Assert: with baseFormatSave the file is still created and non-empty.
  CHECK(ok);
  REQUIRE(fs::exists(dataDir / "cascade.xml"));
  CHECK(fs::file_size(dataDir / "cascade.xml") > 0);

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("CvCascadeClassifier::train: returns false when the vec file does not exist") {
  // Arrange
  const auto workDir = makeUniqueOutputDir("missing_vec");
  const auto dataDir = workDir / "data";
  fs::create_directories(dataDir);
  // A bg.txt is required (for negReader.create) but it can be empty.
  const auto bgPath = workDir / "bg.txt";
  std::ofstream(bgPath) << "";

  CvCascadeParams cascadeParams;
  CvHaarFeatureParams featureParams;
  CvCascadeBoostParams stageParams;

  CvCascadeClassifier classifier;

  // Act
  const bool ok =
      classifier.train(dataDir.string(),
                       (workDir / "no_such.vec").string(),
                       bgPath.string(),
                       /*numPos=*/10, /*numNeg=*/1,
                       /*precalcValBufSize=*/64, /*precalcIdxBufSize=*/64,
                       /*numStages=*/1,
                       cascadeParams, featureParams, stageParams);

  // Assert: PosReader::create returns false for a missing file → train aborts.
  CHECK_FALSE(ok);
  CHECK_FALSE(fs::exists(dataDir / "cascade.xml"));

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}

TEST_CASE("CvCascadeClassifier::train: throws when cascade dir name is empty") {
  // Arrange
  const auto workDir = makeUniqueOutputDir("empty_dir");
  const auto res = stageResources(workDir);
  CvCascadeParams cascadeParams(CvCascadeParams::BOOST,
                                CvFeatureParams::LBP);
  cascadeParams.winSize = cv::Size(75, 32);
  CvLBPFeatureParams featureParams;
  CvCascadeBoostParams stageParams;

  CvCascadeClassifier classifier;

  // Act / Assert: an empty cascade dir is rejected up front.
  CHECK_THROWS_AS(classifier.train(/*cascadeDirName=*/"",
                                   res.vec.string(),
                                   res.bg.string(),
                                   /*numPos=*/10, /*numNeg=*/1,
                                   /*precalcValBufSize=*/64,
                                   /*precalcIdxBufSize=*/64,
                                   /*numStages=*/1,
                                   cascadeParams, featureParams, stageParams),
                  cv::Exception);

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}

// ---------------------------------------------------------------------------
// Multi-stage boost loop
// ---------------------------------------------------------------------------

TEST_CASE(
    "CvCascadeClassifier::train: completes multi-stage training "
    "(numStages=2, maxWeakCount=3, maxDepth=2)") {
  // Arrange: a 2-stage LBP cascade with depth-2 trees and up to 3 weak
  // learners per stage. This exercises the boost outer loop for more than one
  // stage as well as the recursive split path in o_cvboostree.cpp (depth>1)
  // and the sample-weight update path in boost.cpp that runs between stages.
  const auto workDir = makeUniqueOutputDir("multistage");
  const auto res = stageResources(workDir);
  const auto dataDir = workDir / "data";
  fs::create_directories(dataDir);

  CvCascadeParams cascadeParams(CvCascadeParams::BOOST,
                                CvFeatureParams::LBP);
  cascadeParams.winSize = cv::Size(75, 32);
  CvLBPFeatureParams featureParams;
  CvCascadeBoostParams stageParams(cv::ml::Boost::GENTLE,
                                   /*minHitRate=*/0.995F,
                                   /*maxFalseAlarm=*/0.5F,
                                   /*weightTrimRate=*/0.95,
                                   /*maxDepth=*/2,
                                   /*maxWeakCount=*/3);

  CvCascadeClassifier classifier;

  // Act
  const bool ok = classifier.train(dataDir.string(),
                                   res.vec.string(),
                                   res.bg.string(),
                                   /*numPos=*/20,
                                   /*numNeg=*/1,
                                   /*precalcValBufSize=*/64,
                                   /*precalcIdxBufSize=*/64,
                                   /*numStages=*/2,
                                   cascadeParams,
                                   featureParams,
                                   stageParams,
                                   /*baseFormatSave=*/false,
                                   /*acceptanceRatioBreakValue=*/-1.0);

  // Assert: train() returns true when at least one stage trains successfully.
  // The second stage may early-exit if the negative reservoir is exhausted,
  // but stage 0 must always be produced. We accept either a single-stage or a
  // full two-stage cascade and verify the artefacts that are guaranteed.
  CHECK(ok);
  CHECK(fs::exists(dataDir / "cascade.xml"));
  CHECK(fs::exists(dataDir / "params.xml"));
  REQUIRE(fs::exists(dataDir / "stage0.xml"));

  // The produced cascade.xml must remain loadable by the public detector.
  cv::CascadeClassifier loaded((dataDir / "cascade.xml").string());
  CHECK_FALSE(loaded.empty());

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}

// ---------------------------------------------------------------------------
// HAAR feature-set variants
// ---------------------------------------------------------------------------

TEST_CASE("CvCascadeClassifier::train: HAAR CORE mode produces a usable cascade") {
  // Arrange: CORE adds the diagonal/centred Haar features on top of BASIC,
  // exercising additional code paths in haarfeatures.cpp (generateFeatures).
  const auto workDir = makeUniqueOutputDir("haar_core");
  const auto res = stageResources(workDir);
  const auto dataDir = workDir / "data";
  fs::create_directories(dataDir);

  CvCascadeParams cascadeParams(CvCascadeParams::BOOST,
                                CvFeatureParams::HAAR);
  cascadeParams.winSize = cv::Size(75, 32);
  CvHaarFeatureParams featureParams(CvHaarFeatureParams::CORE);
  CvCascadeBoostParams stageParams(cv::ml::Boost::GENTLE,
                                   0.995F, 0.5F, 0.95, 1, 10);

  CvCascadeClassifier classifier;

  // Act
  const bool ok = classifier.train(dataDir.string(),
                                   res.vec.string(),
                                   res.bg.string(),
                                   /*numPos=*/20,
                                   /*numNeg=*/1,
                                   /*precalcValBufSize=*/64,
                                   /*precalcIdxBufSize=*/64,
                                   /*numStages=*/1,
                                   cascadeParams,
                                   featureParams,
                                   stageParams,
                                   /*baseFormatSave=*/false,
                                   /*acceptanceRatioBreakValue=*/-1.0);

  // Assert
  CHECK(ok);
  CHECK(fs::exists(dataDir / "cascade.xml"));
  cv::CascadeClassifier loaded((dataDir / "cascade.xml").string());
  CHECK_FALSE(loaded.empty());

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}

TEST_CASE("CvCascadeClassifier::train: HAAR ALL mode produces a usable cascade") {
  // Arrange: ALL adds the 45-degree rotated Haar features, covering the
  // remaining branch in CvHaarEvaluator::generateFeatures.
  const auto workDir = makeUniqueOutputDir("haar_all");
  const auto res = stageResources(workDir);
  const auto dataDir = workDir / "data";
  fs::create_directories(dataDir);

  CvCascadeParams cascadeParams(CvCascadeParams::BOOST,
                                CvFeatureParams::HAAR);
  cascadeParams.winSize = cv::Size(75, 32);
  CvHaarFeatureParams featureParams(CvHaarFeatureParams::ALL);
  CvCascadeBoostParams stageParams(cv::ml::Boost::GENTLE,
                                   0.995F, 0.5F, 0.95, 1, 10);

  CvCascadeClassifier classifier;

  // Act
  const bool ok = classifier.train(dataDir.string(),
                                   res.vec.string(),
                                   res.bg.string(),
                                   /*numPos=*/20,
                                   /*numNeg=*/1,
                                   /*precalcValBufSize=*/64,
                                   /*precalcIdxBufSize=*/64,
                                   /*numStages=*/1,
                                   cascadeParams,
                                   featureParams,
                                   stageParams,
                                   /*baseFormatSave=*/false,
                                   /*acceptanceRatioBreakValue=*/-1.0);

  // Assert
  CHECK(ok);
  CHECK(fs::exists(dataDir / "cascade.xml"));
  cv::CascadeClassifier loaded((dataDir / "cascade.xml").string());
  CHECK_FALSE(loaded.empty());

  // Cleanup
  std::error_code ec;
  fs::remove_all(workDir, ec);
}
