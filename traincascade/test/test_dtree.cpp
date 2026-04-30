// Direct unit tests for the legacy-style CvDTree decision tree against a
// hand-built CvDTreeTrainData. The cascade trainer uses CvDTree only
// indirectly (via CvCascadeBoostTree), so these tests target the
// CvDTree / CvDTreeTrainData implementation directly with tiny in-memory
// datasets.
//
// All tests follow Arrange / Act / Assert with comments per step.

#include <doctest/doctest.h>

#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/ml.hpp>

#include "o_cvdtree.h"
#include "o_cvdtreenode.h"
#include "o_cvdtreeparams.h"
#include "o_cvdtreetraindata.h"

namespace {

// Helper that builds a 1-D classification dataset:
//   x in [0.0, 0.4] -> class 0
//   x in [0.6, 1.0] -> class 1
// with a clean margin around 0.5. Returns matrices held by the caller.
struct OneDimDataset {
  cv::Mat trainData;   // (N x 1) CV_32F
  cv::Mat responses;   // (N x 1) CV_32F (class labels 0/1)
  cv::Mat varType;     // ((var_count + 1) x 1) CV_8U; 0 = ordered, 1 = categorical
};

OneDimDataset makeOneDimSeparableDataset() {
  // 10 samples, 5 per class, perfectly separable at x = 0.5.
  OneDimDataset d;
  d.trainData = (cv::Mat_<float>(10, 1)
                 << 0.0f, 0.1f, 0.2f, 0.3f, 0.4f,
                    0.6f, 0.7f, 0.8f, 0.9f, 1.0f);
  d.responses = (cv::Mat_<float>(10, 1)
                 << 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1);
  // 1 ordered feature + 1 categorical response.
  d.varType = (cv::Mat_<uchar>(2, 1) << cv::ml::VAR_ORDERED,
                                        cv::ml::VAR_CATEGORICAL);
  return d;
}

// Trains a CvDTree with low-overhead parameters that work for tiny datasets:
// no surrogates, no cross-validation, max_depth = 3, min_sample_count = 1.
CvDTreeParams makeTinyParams() {
  CvDTreeParams p;
  p.max_depth = 3;
  p.min_sample_count = 1;
  p.cv_folds = 0;             // disable internal cross-validation pruning
  p.use_surrogates = false;
  p.use_1se_rule = false;
  p.truncate_pruned_tree = false;
  p.max_categories = 16;
  p.regression_accuracy = 0.0f;
  return p;
}

// Wrap a single feature value into a CV_32F sample row suitable for predict().
cv::Mat sample1D(float value) {
  return (cv::Mat_<float>(1, 1) << value);
}

}  // namespace

// ---------------------------------------------------------------------------
// CvDTree::train(cv::Mat ...) — high-level entry point
// ---------------------------------------------------------------------------

TEST_CASE("CvDTree::train: trains on a 1-D separable dataset and builds a root") {
  // Arrange
  const OneDimDataset d = makeOneDimSeparableDataset();
  const CvDTreeParams params = makeTinyParams();
  CvDTree tree;

  // Act
  const bool trained = tree.train(d.trainData, cv::ml::ROW_SAMPLE,
                                  d.responses, cv::Mat(), cv::Mat(),
                                  d.varType, cv::Mat(), params);

  // Assert
  CHECK(trained);
  REQUIRE(tree.get_root() != nullptr);
  // Every sample reached the root, so its sample_count must equal N.
  CHECK(tree.get_root()->sample_count == d.trainData.rows);
}

TEST_CASE("CvDTree::predict: classifies training points correctly on a 1-D separable dataset") {
  // Arrange
  const OneDimDataset d = makeOneDimSeparableDataset();
  CvDTree tree;
  REQUIRE(tree.train(d.trainData, cv::ml::ROW_SAMPLE, d.responses,
                     cv::Mat(), cv::Mat(), d.varType, cv::Mat(),
                     makeTinyParams()));

  // Act / Assert: every training row must come back with its own label.
  bool allCorrect = true;
  for (int i = 0; i < d.trainData.rows; ++i) {
    const cv::Mat row = d.trainData.row(i);
    const CvDTreeNode* leaf = tree.predict(row);
    if (leaf == nullptr) {
      allCorrect = false;
      break;
    }
    const float expected = d.responses.at<float>(i, 0);
    if (cvRound(leaf->value) != cvRound(expected)) {
      allCorrect = false;
      break;
    }
  }
  CHECK(allCorrect);
}

TEST_CASE("CvDTree::predict: generalizes to held-out points either side of the margin") {
  // Arrange
  const OneDimDataset d = makeOneDimSeparableDataset();
  CvDTree tree;
  REQUIRE(tree.train(d.trainData, cv::ml::ROW_SAMPLE, d.responses,
                     cv::Mat(), cv::Mat(), d.varType, cv::Mat(),
                     makeTinyParams()));

  // Act
  const CvDTreeNode* lowLeaf = tree.predict(sample1D(0.05f));
  const CvDTreeNode* highLeaf = tree.predict(sample1D(0.95f));

  // Assert
  REQUIRE(lowLeaf != nullptr);
  REQUIRE(highLeaf != nullptr);
  CHECK(cvRound(lowLeaf->value) == 0);
  CHECK(cvRound(highLeaf->value) == 1);
}

TEST_CASE("CvDTree::clear: drops the tree and frees the trained state") {
  // Arrange
  const OneDimDataset d = makeOneDimSeparableDataset();
  CvDTree tree;
  REQUIRE(tree.train(d.trainData, cv::ml::ROW_SAMPLE, d.responses,
                     cv::Mat(), cv::Mat(), d.varType, cv::Mat(),
                     makeTinyParams()));
  REQUIRE(tree.get_root() != nullptr);

  // Act
  tree.clear();

  // Assert
  CHECK(tree.get_root() == nullptr);
}

TEST_CASE("CvDTree: re-training on the same instance produces a fresh tree") {
  // Arrange
  const OneDimDataset d = makeOneDimSeparableDataset();
  CvDTree tree;
  REQUIRE(tree.train(d.trainData, cv::ml::ROW_SAMPLE, d.responses,
                     cv::Mat(), cv::Mat(), d.varType, cv::Mat(),
                     makeTinyParams()));
  const CvDTreeNode* firstRoot = tree.get_root();
  REQUIRE(firstRoot != nullptr);

  // Act: train on a different dataset (responses inverted).
  cv::Mat invertedResponses = d.responses.clone();
  for (int i = 0; i < invertedResponses.rows; ++i) {
    invertedResponses.at<float>(i, 0) = 1.0f - invertedResponses.at<float>(i, 0);
  }
  REQUIRE(tree.train(d.trainData, cv::ml::ROW_SAMPLE, invertedResponses,
                     cv::Mat(), cv::Mat(), d.varType, cv::Mat(),
                     makeTinyParams()));

  // Assert: the inverted training set must classify x=0.05 as class 1 now.
  const CvDTreeNode* leaf = tree.predict(sample1D(0.05f));
  REQUIRE(leaf != nullptr);
  CHECK(cvRound(leaf->value) == 1);
}

// ---------------------------------------------------------------------------
// CvDTree::train(CvDTreeTrainData*) — the shared-data overload that
// CvBoostTree / CvCascadeBoostTree actually use.
// ---------------------------------------------------------------------------

TEST_CASE("CvDTree::train(CvDTreeTrainData*): trains successfully against an externally-built data object") {
  // Arrange: build a CvDTreeTrainData manually, then pass it to CvDTree.
  const OneDimDataset d = makeOneDimSeparableDataset();
  const CvDTreeParams params = makeTinyParams();
  CvMat trainHdr = cvMat(d.trainData);
  CvMat respHdr = cvMat(d.responses);
  CvMat vtypeHdr = cvMat(d.varType);

  CvDTreeTrainData data(&trainHdr, cv::ml::ROW_SAMPLE, &respHdr,
                        /*varIdx=*/nullptr, /*sampleIdx=*/nullptr,
                        &vtypeHdr, /*missingDataMask=*/nullptr,
                        params, /*shared=*/true, /*addLabels=*/false);

  CvDTree tree;

  // Act
  const bool trained = tree.train(&data, /*subsampleIdx=*/nullptr);

  // Assert
  CHECK(trained);
  REQUIRE(tree.get_root() != nullptr);
  CHECK(tree.get_root()->sample_count == d.trainData.rows);

  const CvDTreeNode* leaf = tree.predict(sample1D(0.95f));
  REQUIRE(leaf != nullptr);
  CHECK(cvRound(leaf->value) == 1);
}

// ---------------------------------------------------------------------------
// CvDTreeTrainData direct API
// ---------------------------------------------------------------------------

TEST_CASE("CvDTreeTrainData: reports classifier mode and var counts after set_data") {
  // Arrange
  const OneDimDataset d = makeOneDimSeparableDataset();
  CvMat trainHdr = cvMat(d.trainData);
  CvMat respHdr = cvMat(d.responses);
  CvMat vtypeHdr = cvMat(d.varType);
  CvDTreeTrainData data;

  // Act
  data.set_data(&trainHdr, cv::ml::ROW_SAMPLE, &respHdr,
                /*varIdx=*/nullptr, /*sampleIdx=*/nullptr,
                &vtypeHdr, /*missingDataMask=*/nullptr,
                makeTinyParams(), /*shared=*/false, /*addLabels=*/false);

  // Assert
  CHECK(data.is_classifier);
  CHECK(data.sample_count == 10);
  CHECK(data.var_count == 1);
  // Categorical response -> two classes (0 and 1).
  CHECK(data.get_num_classes() == 2);
  // The single ordered feature should report a non-categorical type
  // (negative when unwrapped via get_var_type).
  CHECK(data.get_var_type(0) < 0);
  // clear() releases internal storage without crashing.
  data.clear();
}

TEST_CASE("CvDTreeTrainData::clear: is safe to call on a default-constructed instance") {
  // Arrange
  CvDTreeTrainData data;

  // Act / Assert: must not crash, must not throw.
  data.clear();
  CHECK(data.sample_count == 0);
  CHECK(data.var_count == 0);
}
