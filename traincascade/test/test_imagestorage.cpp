#include <doctest/doctest.h>

#include <opencv2/core.hpp>

#include "imagestorage.h"

TEST_CASE("CvCascadeImageReader::create: returns false when positive vec file is missing") {
  // Arrange
  CvCascadeImageReader reader;

  // Act
  const bool ok = reader.create("/no/such/file.vec",
                                "/no/such/bg.txt",
                                cv::Size(24, 24));

  // Assert
  CHECK_FALSE(ok);
}

TEST_CASE("CvCascadeImageReader::create: returns false even with a single missing file") {
  // Arrange
  CvCascadeImageReader reader;

  // Act: same nonexistent path on both arguments — both readers must fail.
  const bool ok = reader.create("",
                                "",
                                cv::Size(24, 24));

  // Assert
  CHECK_FALSE(ok);
}
