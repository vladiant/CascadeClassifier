/**
 * @file o_blockedrange.h
 * @brief Lightweight serial replacement for OpenCV / TBB parallel primitives.
 *
 * The cascade trainer was originally written against TBB's
 * @c blocked_range / @c parallel_for / @c parallel_reduce. This header
 * provides a serial drop-in so the library can build without a parallel
 * runtime; the @c BlockedRange object simply describes a half-open
 * integer interval [begin, end) and the @c parallel_* helpers run the
 * supplied @c Body in the calling thread.
 */
#pragma once

/// Half-open integer range used as the iteration domain for @ref parallel_for / @ref parallel_reduce.
class BlockedRange {
 public:
  BlockedRange() : _begin(0), _end(0), _grainsize(0) {}
  BlockedRange(int b, int e, int g = 1) : _begin(b), _end(e), _grainsize(g) {}
  int begin() const { return _begin; }
  int end() const { return _end; }
  /// Approximate chunk size used by parallel runtimes; ignored by this serial impl.
  int grainsize() const { return _grainsize; }

 protected:
  int _begin, _end, _grainsize;
};

/**
 * @brief Serial replacement for @c tbb::parallel_for: invokes @p body once
 *        with the full @p range.
 */
template <typename Body>
static inline void parallel_for(const BlockedRange& range, const Body& body) {
  body(range);
}

/// Placeholder type kept for API compatibility with TBB's @c split tag.
class Split {};

/**
 * @brief Serial replacement for @c tbb::parallel_reduce: invokes @p body
 *        once with the full @p range and never calls @c join.
 */
template <typename Body>
static inline void parallel_reduce(const BlockedRange& range, Body& body) {
  body(range);
}
