/**
 * @file o_cvstatmodel.h
 * @brief Minimal base class extracted from OpenCV's legacy @c CvStatModel.
 *
 * @ref CvStatModel is the root of the legacy ML class hierarchy reused by
 * the cascade trainer (CvDTree -> CvBoost -> CvCascadeBoost). It only
 * provides a virtual destructor and a @ref clear hook so subclasses can
 * release their internal buffers polymorphically.
 */
#pragma once

/**
 * @brief Polymorphic base for the legacy ML model hierarchy.
 *
 * Direct subclasses in this project: @c CvDTree (decision tree) and
 * @c CvBoost (boosted ensemble). The @c default_model_name field is
 * preserved for ABI compatibility with OpenCV's stored XML format.
 */
class CvStatModel {
 public:
  CvStatModel();
  virtual ~CvStatModel();

  /// Release internal state so the instance can be retrained or destroyed.
  virtual void clear();

 protected:
  const char* default_model_name; ///< XML tag used by OpenCV's legacy persistence layer.
};
