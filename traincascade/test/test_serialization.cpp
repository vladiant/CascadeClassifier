// Round-trip serialization tests for the *Params types.
//
// The cascade trainer writes a `params.xml` next to each cascade stage and
// reads it back when resuming training or saving the final cascade. These
// tests verify that every *Params type that participates in that workflow
// survives a `cv::FileStorage` write -> read round trip without loss.
//
// All tests use in-memory FileStorage so nothing touches the filesystem.

#include <doctest/doctest.h>

#include <opencv2/core.hpp>
#include <opencv2/ml/ml.hpp>

#include "boost.h"
#include "cascadeclassifier.h"
#include "haarfeatures.h"
#include "lbpfeatures.h"
#include "traincascade_features.h"

namespace {

// Serialize `src` into an in-memory XML buffer under a wrapping section named
// `section` and return the buffer.
template <typename Params>
std::string writeToMemory(const Params& src, const std::string& section) {
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << section << "{";
  src.write(fs);
  fs << "}";
  return fs.releaseAndGetString();
}

// Deserialize a previously-written buffer into `dst` using the same wrapping
// section name. Returns the value reported by `dst.read(...)`.
template <typename Params>
bool readFromMemory(Params& dst, const std::string& xml,
                    const std::string& section) {
  cv::FileStorage fs(xml, cv::FileStorage::READ | cv::FileStorage::MEMORY);
  return dst.read(fs[section]);
}

}  // namespace

// ---------------------------------------------------------------------------
// CvCascadeParams
// ---------------------------------------------------------------------------

TEST_CASE("CvCascadeParams: round-trips through FileStorage (HAAR)") {
  // Arrange: configure a non-default cascade params instance.
  CvCascadeParams src(CvCascadeParams::BOOST, CvFeatureParams::HAAR);
  src.winSize = cv::Size(48, 24);

  // Act: write to memory and read it back into a fresh instance.
  const std::string xml = writeToMemory(src, "cascadeParams");
  CvCascadeParams dst;
  const bool ok = readFromMemory(dst, xml, "cascadeParams");

  // Assert
  CHECK(ok);
  CHECK(dst.stageType == src.stageType);
  CHECK(dst.featureType == src.featureType);
  CHECK(dst.winSize.width == src.winSize.width);
  CHECK(dst.winSize.height == src.winSize.height);
}

TEST_CASE("CvCascadeParams: round-trips through FileStorage (LBP)") {
  // Arrange
  CvCascadeParams src(CvCascadeParams::BOOST, CvFeatureParams::LBP);
  src.winSize = cv::Size(75, 32);

  // Act
  const std::string xml = writeToMemory(src, "cascadeParams");
  CvCascadeParams dst;
  const bool ok = readFromMemory(dst, xml, "cascadeParams");

  // Assert
  CHECK(ok);
  CHECK(dst.featureType == CvFeatureParams::LBP);
  CHECK(dst.winSize == cv::Size(75, 32));
}

TEST_CASE("CvCascadeParams: round-trips through FileStorage (HOG)") {
  // Arrange
  CvCascadeParams src(CvCascadeParams::BOOST, CvFeatureParams::HOG);
  src.winSize = cv::Size(64, 128);

  // Act
  const std::string xml = writeToMemory(src, "cascadeParams");
  CvCascadeParams dst;
  const bool ok = readFromMemory(dst, xml, "cascadeParams");

  // Assert
  CHECK(ok);
  CHECK(dst.featureType == CvFeatureParams::HOG);
  CHECK(dst.winSize == cv::Size(64, 128));
}

TEST_CASE("CvCascadeParams::read returns false for an empty node") {
  // Arrange: build a FileStorage that does not contain the expected section.
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "other" << 1;
  const std::string xml = fs.releaseAndGetString();

  cv::FileStorage rs(xml, cv::FileStorage::READ | cv::FileStorage::MEMORY);
  CvCascadeParams dst;

  // Act
  const bool ok = dst.read(rs["cascadeParams"]);

  // Assert
  CHECK_FALSE(ok);
}

TEST_CASE("CvCascadeParams::read rejects non-positive window sizes") {
  // Arrange: hand-craft an XML payload with width=0.
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "cascadeParams"
     << "{"
     << "stageType" << "BOOST"
     << "featureType" << "HAAR"
     << "height" << 24
     << "width" << 0
     << "}";
  const std::string xml = fs.releaseAndGetString();

  CvCascadeParams dst;

  // Act
  const bool ok = readFromMemory(dst, xml, "cascadeParams");

  // Assert
  CHECK_FALSE(ok);
}

TEST_CASE("CvCascadeParams::read rejects unknown feature type strings") {
  // Arrange
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "cascadeParams"
     << "{"
     << "stageType" << "BOOST"
     << "featureType" << "UNKNOWN"
     << "height" << 24
     << "width" << 24
     << "}";
  const std::string xml = fs.releaseAndGetString();

  CvCascadeParams dst;

  // Act
  const bool ok = readFromMemory(dst, xml, "cascadeParams");

  // Assert
  CHECK_FALSE(ok);
}

// ---------------------------------------------------------------------------
// CvFeatureParams (base class)
// ---------------------------------------------------------------------------

TEST_CASE("CvFeatureParams: round-trips maxCatCount and featSize") {
  // Arrange
  CvFeatureParams src;
  src.maxCatCount = 42;
  src.featSize = 7;

  // Act
  const std::string xml = writeToMemory(src, "featureParams");
  CvFeatureParams dst;
  const bool ok = readFromMemory(dst, xml, "featureParams");

  // Assert
  CHECK(ok);
  CHECK(dst.maxCatCount == 42);
  CHECK(dst.featSize == 7);
}

TEST_CASE("CvFeatureParams::read returns false for an empty node") {
  // Arrange
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "other" << 1;
  const std::string xml = fs.releaseAndGetString();

  cv::FileStorage rs(xml, cv::FileStorage::READ | cv::FileStorage::MEMORY);
  CvFeatureParams dst;

  // Act
  const bool ok = dst.read(rs["featureParams"]);

  // Assert
  CHECK_FALSE(ok);
}

// ---------------------------------------------------------------------------
// CvHaarFeatureParams (extends CvFeatureParams with `mode`)
// ---------------------------------------------------------------------------

TEST_CASE("CvHaarFeatureParams: round-trips BASIC mode") {
  // Arrange
  CvHaarFeatureParams src(CvHaarFeatureParams::BASIC);
  src.maxCatCount = 5;
  src.featSize = 1;

  // Act
  const std::string xml = writeToMemory(src, "haarFeatureParams");
  CvHaarFeatureParams dst(CvHaarFeatureParams::ALL);  // start from a different mode
  const bool ok = readFromMemory(dst, xml, "haarFeatureParams");

  // Assert
  CHECK(ok);
  CHECK(dst.mode == CvHaarFeatureParams::BASIC);
  CHECK(dst.maxCatCount == 5);
  CHECK(dst.featSize == 1);
}

TEST_CASE("CvHaarFeatureParams: round-trips CORE mode") {
  // Arrange
  CvHaarFeatureParams src(CvHaarFeatureParams::CORE);

  // Act
  const std::string xml = writeToMemory(src, "haarFeatureParams");
  CvHaarFeatureParams dst;
  const bool ok = readFromMemory(dst, xml, "haarFeatureParams");

  // Assert
  CHECK(ok);
  CHECK(dst.mode == CvHaarFeatureParams::CORE);
}

TEST_CASE("CvHaarFeatureParams: round-trips ALL mode") {
  // Arrange
  CvHaarFeatureParams src(CvHaarFeatureParams::ALL);

  // Act
  const std::string xml = writeToMemory(src, "haarFeatureParams");
  CvHaarFeatureParams dst;
  const bool ok = readFromMemory(dst, xml, "haarFeatureParams");

  // Assert
  CHECK(ok);
  CHECK(dst.mode == CvHaarFeatureParams::ALL);
}

TEST_CASE("CvHaarFeatureParams::read rejects unknown mode strings") {
  // Arrange: payload with a valid CvFeatureParams body but a bogus mode.
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "haarFeatureParams"
     << "{"
     << "maxCatCount" << 0
     << "featSize" << 1
     << "mode" << "BOGUS"
     << "}";
  const std::string xml = fs.releaseAndGetString();

  CvHaarFeatureParams dst;

  // Act
  const bool ok = readFromMemory(dst, xml, "haarFeatureParams");

  // Assert
  CHECK_FALSE(ok);
}

TEST_CASE("CvHaarFeatureParams::read returns false when mode is missing") {
  // Arrange
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "haarFeatureParams"
     << "{"
     << "maxCatCount" << 0
     << "featSize" << 1
     << "}";
  const std::string xml = fs.releaseAndGetString();

  CvHaarFeatureParams dst;

  // Act
  const bool ok = readFromMemory(dst, xml, "haarFeatureParams");

  // Assert
  CHECK_FALSE(ok);
}

// ---------------------------------------------------------------------------
// CvLBPFeatureParams (uses the base CvFeatureParams read/write directly)
// ---------------------------------------------------------------------------

TEST_CASE("CvLBPFeatureParams: round-trips through FileStorage") {
  // Arrange: defaults are maxCatCount=256, featSize=1.
  CvLBPFeatureParams src;
  REQUIRE(src.maxCatCount == 256);
  REQUIRE(src.featSize == 1);

  // Act
  const std::string xml = writeToMemory(src, "lbpFeatureParams");
  CvLBPFeatureParams dst;
  dst.maxCatCount = 0;  // ensure reading actually overwrites the values
  dst.featSize = 0;
  const bool ok = readFromMemory(dst, xml, "lbpFeatureParams");

  // Assert
  CHECK(ok);
  CHECK(dst.maxCatCount == 256);
  CHECK(dst.featSize == 1);
}

// ---------------------------------------------------------------------------
// CvCascadeBoostParams
// ---------------------------------------------------------------------------

TEST_CASE("CvCascadeBoostParams: round-trips GENTLE boost configuration") {
  // Arrange
  CvCascadeBoostParams src(cv::ml::Boost::GENTLE,
                           /*minHitRate=*/0.995F,
                           /*maxFalseAlarm=*/0.5F,
                           /*weightTrimRate=*/0.95,
                           /*maxDepth=*/3,
                           /*maxWeakCount=*/100);

  // Act
  const std::string xml = writeToMemory(src, "stageParams");
  CvCascadeBoostParams dst;
  const bool ok = readFromMemory(dst, xml, "stageParams");

  // Assert
  CHECK(ok);
  CHECK(dst.boost_type == cv::ml::Boost::GENTLE);
  CHECK(dst.minHitRate == doctest::Approx(0.995F));
  CHECK(dst.maxFalseAlarm == doctest::Approx(0.5F));
  CHECK(dst.weight_trim_rate == doctest::Approx(0.95));
  CHECK(dst.max_depth == 3);
  CHECK(dst.weak_count == 100);
}

TEST_CASE("CvCascadeBoostParams: round-trips DISCRETE boost configuration") {
  // Arrange: the parameterized ctor unconditionally sets boost_type to
  // GENTLE, so we override it directly to actually exercise DISCRETE.
  CvCascadeBoostParams src(cv::ml::Boost::DISCRETE, 0.99F, 0.4F, 0.9, 1, 50);
  src.boost_type = cv::ml::Boost::DISCRETE;

  // Act
  const std::string xml = writeToMemory(src, "stageParams");
  CvCascadeBoostParams dst;
  const bool ok = readFromMemory(dst, xml, "stageParams");

  // Assert
  CHECK(ok);
  CHECK(dst.boost_type == cv::ml::Boost::DISCRETE);
  CHECK(dst.minHitRate == doctest::Approx(0.99F));
  CHECK(dst.maxFalseAlarm == doctest::Approx(0.4F));
  CHECK(dst.max_depth == 1);
  CHECK(dst.weak_count == 50);
}

TEST_CASE("CvCascadeBoostParams: round-trips REAL and LOGIT boost types") {
  // Arrange: see DISCRETE case for why boost_type must be set directly.
  CvCascadeBoostParams realSrc(cv::ml::Boost::REAL, 0.9F, 0.5F, 0.8, 2, 10);
  realSrc.boost_type = cv::ml::Boost::REAL;
  CvCascadeBoostParams logitSrc(cv::ml::Boost::LOGIT, 0.9F, 0.5F, 0.8, 2, 10);
  logitSrc.boost_type = cv::ml::Boost::LOGIT;

  // Act
  CvCascadeBoostParams realDst;
  CvCascadeBoostParams logitDst;
  const bool okReal = readFromMemory(
      realDst, writeToMemory(realSrc, "stageParams"), "stageParams");
  const bool okLogit = readFromMemory(
      logitDst, writeToMemory(logitSrc, "stageParams"), "stageParams");

  // Assert
  CHECK(okReal);
  CHECK(okLogit);
  CHECK(realDst.boost_type == cv::ml::Boost::REAL);
  CHECK(logitDst.boost_type == cv::ml::Boost::LOGIT);
}

TEST_CASE("CvCascadeBoostParams::read throws on out-of-range values") {
  // Arrange: minHitRate=0 is rejected by the implementation.
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "stageParams"
     << "{"
     << "boostType" << "GAB"
     << "minHitRate" << 0.0F
     << "maxFalseAlarm" << 0.5F
     << "weightTrimRate" << 0.95
     << "maxDepth" << 1
     << "maxWeakCount" << 10
     << "}";
  const std::string xml = fs.releaseAndGetString();

  CvCascadeBoostParams dst;

  // Act / Assert
  CHECK_THROWS_AS(readFromMemory(dst, xml, "stageParams"), cv::Exception);
}

TEST_CASE("CvCascadeBoostParams::read throws on unknown boost type") {
  // Arrange
  cv::FileStorage fs(".xml",
                     cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  fs << "stageParams"
     << "{"
     << "boostType" << "BOGUS"
     << "minHitRate" << 0.99F
     << "maxFalseAlarm" << 0.5F
     << "weightTrimRate" << 0.95
     << "maxDepth" << 1
     << "maxWeakCount" << 10
     << "}";
  const std::string xml = fs.releaseAndGetString();

  CvCascadeBoostParams dst;

  // Act / Assert
  CHECK_THROWS_AS(readFromMemory(dst, xml, "stageParams"), cv::Exception);
}
