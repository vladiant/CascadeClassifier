/**
 * @file o_utils.h
 * @brief Small utility predicates and helpers reused by the legacy ML code.
 */
#pragma once

struct CvMat;

/**
 * @brief Indirect comparator: compares @c arr[a] with @c arr[b].
 *
 * Useful with @c std::sort to sort an array of indices by the values
 * they reference, which is the standard pattern used by the decision-
 * tree splitter to order ordered-variable candidates without permuting
 * the underlying value array.
 */
template <typename T, typename Idx>
class LessThanIdx {
 public:
  explicit LessThanIdx(const T* _arr) : arr(_arr) {}
  bool operator()(Idx a, Idx b) const { return arr[a] < arr[b]; }
  const T* arr;
};

/// Indirect comparator: compares the values pointed to by @p a and @p b.
template <typename T>
class LessThanPtr {
 public:
  bool operator()(T* a, T* b) const { return *a < *b; }
};

/// `qsort` callback that compares two integers stored at @p a and @p b.
int icvCmpIntegers(const void* a, const void* b);

/// Round @p size up to the next multiple of @p align (power of two).
int cvAlign(int size, int align);

/// Probe a categorical-split bitmask: returns 0 (left) or 1 (right) for category @p idx.
int CV_DTREE_CAT_DIR(int idx, const int* subset);

/**
 * @brief Validate and normalize an external index array.
 *
 * Used by the trainer to pre-process the optional @c sampleIdx /
 * @c varIdx arguments before feeding them to the data structures.
 * Optionally raises an exception on duplicate entries.
 */
CvMat* cvPreprocessIndexArray(const CvMat* idx_arr, int data_arr_size,
                              bool check_for_duplicates = false);
