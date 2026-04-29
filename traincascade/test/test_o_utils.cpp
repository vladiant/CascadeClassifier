#include <doctest/doctest.h>

#include <algorithm>
#include <climits>
#include <vector>

#include <opencv2/core/core_c.h>
#include <opencv2/core.hpp>

#include "o_utils.h"
#include "o_blockedrange.h"
#include "o_cvdtreenode.h"
#include "o_cvdtreesplit.h"

// ---------------------------------------------------------------------------
// icvCmpIntegers
// ---------------------------------------------------------------------------

TEST_CASE("icvCmpIntegers: compares two integers via void pointers") {
  // Arrange
  const int smaller = 1;
  const int larger = 5;
  const int equal_a = 7;
  const int equal_b = 7;

  // Act
  const int neg = icvCmpIntegers(&smaller, &larger);
  const int pos = icvCmpIntegers(&larger, &smaller);
  const int zero = icvCmpIntegers(&equal_a, &equal_b);

  // Assert
  CHECK(neg < 0);
  CHECK(pos > 0);
  CHECK(zero == 0);
}

TEST_CASE("icvCmpIntegers: works as comparator for qsort") {
  // Arrange
  std::vector<int> values = {5, 2, 9, 1, 7, 3};

  // Act
  std::qsort(values.data(), values.size(), sizeof(int), icvCmpIntegers);

  // Assert
  CHECK(std::is_sorted(values.begin(), values.end()));
}

// ---------------------------------------------------------------------------
// cvAlign
// ---------------------------------------------------------------------------

TEST_CASE("cvAlign: aligns to power-of-two boundaries") {
  // Arrange / Act / Assert
  CHECK(cvAlign(0, 8) == 0);
  CHECK(cvAlign(1, 8) == 8);
  CHECK(cvAlign(7, 8) == 8);
  CHECK(cvAlign(8, 8) == 8);
  CHECK(cvAlign(9, 8) == 16);
  CHECK(cvAlign(15, 16) == 16);
  CHECK(cvAlign(16, 16) == 16);
  CHECK(cvAlign(17, 16) == 32);
}

TEST_CASE("cvAlign: align == 1 acts as identity") {
  // Arrange / Act / Assert
  CHECK(cvAlign(0, 1) == 0);
  CHECK(cvAlign(1, 1) == 1);
  CHECK(cvAlign(42, 1) == 42);
  CHECK(cvAlign(123, 1) == 123);
}

// ---------------------------------------------------------------------------
// CV_DTREE_CAT_DIR
// ---------------------------------------------------------------------------

TEST_CASE("CV_DTREE_CAT_DIR: returns +1 for unset bits and -1 for set bits") {
  // Arrange: a 64-bit subset (two 32-bit words) with a known bit pattern.
  // Bit 0 set, bit 1 unset, bit 5 set, bit 32 set, bit 33 unset.
  int subset[2] = {0, 0};
  subset[0] = (1 << 0) | (1 << 5);
  subset[1] = (1 << 0); // bit 32 of the 64-bit "view"

  // Act / Assert
  CHECK(CV_DTREE_CAT_DIR(0, subset) == -1);
  CHECK(CV_DTREE_CAT_DIR(1, subset) == 1);
  CHECK(CV_DTREE_CAT_DIR(2, subset) == 1);
  CHECK(CV_DTREE_CAT_DIR(5, subset) == -1);
  CHECK(CV_DTREE_CAT_DIR(31, subset) == 1);
  CHECK(CV_DTREE_CAT_DIR(32, subset) == -1);
  CHECK(CV_DTREE_CAT_DIR(33, subset) == 1);
}

// ---------------------------------------------------------------------------
// LessThanIdx / LessThanPtr
// ---------------------------------------------------------------------------

TEST_CASE("LessThanIdx: orders indices by referenced array values") {
  // Arrange
  const float arr[] = {3.0f, 1.0f, 4.0f, 1.5f, 2.0f};
  std::vector<int> idx = {0, 1, 2, 3, 4};

  // Act
  std::sort(idx.begin(), idx.end(), LessThanIdx<float, int>(arr));

  // Assert: indices sorted such that arr[idx[i]] is non-decreasing
  for (std::size_t i = 1; i < idx.size(); ++i) {
    CHECK(arr[idx[i - 1]] <= arr[idx[i]]);
  }
}

TEST_CASE("LessThanIdx: handles single element and equal values") {
  // Arrange
  const int singletonArr[] = {42};
  std::vector<int> singletonIdx = {0};
  const int equalArr[] = {7, 7, 7};
  std::vector<int> equalIdx = {0, 1, 2};

  // Act
  std::sort(singletonIdx.begin(), singletonIdx.end(),
            LessThanIdx<int, int>(singletonArr));
  std::sort(equalIdx.begin(), equalIdx.end(),
            LessThanIdx<int, int>(equalArr));

  // Assert
  CHECK(singletonIdx.size() == 1);
  CHECK(singletonIdx[0] == 0);
  CHECK(equalIdx.size() == 3); // stable order not required, just a no-throw run
}

TEST_CASE("LessThanPtr: orders pointers by their dereferenced values") {
  // Arrange
  int a = 10;
  int b = 5;
  int c = 7;
  std::vector<int*> ptrs = {&a, &b, &c};

  // Act
  std::sort(ptrs.begin(), ptrs.end(), LessThanPtr<int>());

  // Assert
  CHECK(*ptrs[0] == 5);
  CHECK(*ptrs[1] == 7);
  CHECK(*ptrs[2] == 10);
}

// ---------------------------------------------------------------------------
// cvPreprocessIndexArray
// ---------------------------------------------------------------------------

TEST_CASE("cvPreprocessIndexArray: copies a sorted CV_32SC1 index array") {
  // Arrange
  int data[] = {0, 2, 4};
  CvMat m = cvMat(1, 3, CV_32SC1, data);

  // Act
  CvMat* out = cvPreprocessIndexArray(&m, /*data_arr_size=*/5);

  // Assert
  REQUIRE(out != nullptr);
  CHECK(CV_MAT_TYPE(out->type) == CV_32SC1);
  CHECK(out->cols == 3);
  CHECK(out->data.i[0] == 0);
  CHECK(out->data.i[1] == 2);
  CHECK(out->data.i[2] == 4);

  cvReleaseMat(&out);
}

TEST_CASE("cvPreprocessIndexArray: expands an 8U mask into selected indices") {
  // Arrange: mask with three set entries out of five.
  unsigned char mask[] = {1, 0, 1, 0, 1};
  CvMat m = cvMat(1, 5, CV_8UC1, mask);

  // Act
  CvMat* out = cvPreprocessIndexArray(&m, /*data_arr_size=*/5);

  // Assert
  REQUIRE(out != nullptr);
  CHECK(out->cols == 3);
  CHECK(out->data.i[0] == 0);
  CHECK(out->data.i[1] == 2);
  CHECK(out->data.i[2] == 4);

  cvReleaseMat(&out);
}

TEST_CASE("cvPreprocessIndexArray: throws when no element is selected by mask") {
  // Arrange
  unsigned char mask[] = {0, 0, 0};
  CvMat m = cvMat(1, 3, CV_8UC1, mask);

  // Act / Assert
  CHECK_THROWS_AS(cvPreprocessIndexArray(&m, 3), cv::Exception);
}

TEST_CASE("cvPreprocessIndexArray: throws on out-of-range integer index") {
  // Arrange
  int data[] = {0, 1, 99};
  CvMat m = cvMat(1, 3, CV_32SC1, data);

  // Act / Assert
  CHECK_THROWS_AS(cvPreprocessIndexArray(&m, /*data_arr_size=*/5),
                  cv::Exception);
}

TEST_CASE("cvPreprocessIndexArray: throws on duplicates when checking enabled") {
  // Arrange
  int data[] = {1, 2, 2, 3};
  CvMat m = cvMat(1, 4, CV_32SC1, data);

  // Act / Assert
  CHECK_THROWS_AS(cvPreprocessIndexArray(&m, /*data_arr_size=*/10,
                                         /*check_for_duplicates=*/true),
                  cv::Exception);
}

TEST_CASE("cvPreprocessIndexArray: throws when index array is multi-dimensional") {
  // Arrange
  int data[] = {0, 1, 2, 3};
  CvMat m = cvMat(2, 2, CV_32SC1, data);

  // Act / Assert
  CHECK_THROWS_AS(cvPreprocessIndexArray(&m, 4), cv::Exception);
}

TEST_CASE("cvPreprocessIndexArray: throws on unsupported data type") {
  // Arrange
  float data[] = {0.0f, 1.0f, 2.0f};
  CvMat m = cvMat(1, 3, CV_32FC1, data);

  // Act / Assert
  CHECK_THROWS_AS(cvPreprocessIndexArray(&m, 5), cv::Exception);
}

// ---------------------------------------------------------------------------
// BlockedRange / parallel_for / parallel_reduce
// ---------------------------------------------------------------------------

TEST_CASE("BlockedRange: default-constructed range is empty") {
  // Arrange / Act
  BlockedRange r;

  // Assert
  CHECK(r.begin() == 0);
  CHECK(r.end() == 0);
  CHECK(r.grainsize() == 0);
}

TEST_CASE("BlockedRange: explicit construction stores bounds and grainsize") {
  // Arrange / Act
  BlockedRange r(5, 17, 4);

  // Assert
  CHECK(r.begin() == 5);
  CHECK(r.end() == 17);
  CHECK(r.grainsize() == 4);
}

TEST_CASE("BlockedRange: default grainsize is 1") {
  // Arrange / Act
  BlockedRange r(0, 10);

  // Assert
  CHECK(r.grainsize() == 1);
}

namespace {
struct SumBody {
  mutable long long sum = 0;
  void operator()(const BlockedRange& r) const {
    for (int i = r.begin(); i < r.end(); ++i) {
      sum += i;
    }
  }
};
} // namespace

TEST_CASE("parallel_for: invokes body once over the whole range") {
  // Arrange
  SumBody body;
  BlockedRange r(1, 11);

  // Act
  parallel_for(r, body);

  // Assert: 1 + 2 + ... + 10 == 55
  CHECK(body.sum == 55);
}

TEST_CASE("parallel_for: empty range does not invoke the loop body") {
  // Arrange
  SumBody body;
  BlockedRange r(7, 7);

  // Act
  parallel_for(r, body);

  // Assert
  CHECK(body.sum == 0);
}

TEST_CASE("parallel_reduce: forwards range to body and mutates it") {
  // Arrange
  SumBody body;
  BlockedRange r(0, 5);

  // Act
  parallel_reduce(r, body);

  // Assert: 0 + 1 + 2 + 3 + 4 == 10
  CHECK(body.sum == 10);
}

// ---------------------------------------------------------------------------
// CvDTreeNode helpers
// ---------------------------------------------------------------------------

TEST_CASE("CvDTreeNode::get_num_valid: falls back to sample_count when null") {
  // Arrange
  CvDTreeNode node{};
  node.sample_count = 42;
  node.num_valid = nullptr;

  // Act / Assert
  CHECK(node.get_num_valid(0) == 42);
  CHECK(node.get_num_valid(99) == 42);
}

TEST_CASE("CvDTreeNode::get_num_valid: reads from num_valid array when present") {
  // Arrange
  int valid[] = {3, 7, 11};
  CvDTreeNode node{};
  node.sample_count = 0;
  node.num_valid = valid;

  // Act / Assert
  CHECK(node.get_num_valid(0) == 3);
  CHECK(node.get_num_valid(1) == 7);
  CHECK(node.get_num_valid(2) == 11);
}

TEST_CASE("CvDTreeNode::set_num_valid: writes to array when allocated") {
  // Arrange
  int valid[] = {0, 0, 0};
  CvDTreeNode node{};
  node.num_valid = valid;

  // Act
  node.set_num_valid(1, 99);

  // Assert
  CHECK(valid[0] == 0);
  CHECK(valid[1] == 99);
  CHECK(valid[2] == 0);
}

TEST_CASE("CvDTreeNode::set_num_valid: is a no-op when num_valid is null") {
  // Arrange
  CvDTreeNode node{};
  node.num_valid = nullptr;
  node.sample_count = 5;

  // Act / Assert (must not crash)
  node.set_num_valid(0, 123);
  CHECK(node.get_num_valid(0) == 5);
}
