#pragma once

// DTreeBestSplitFinder
class BlockedRange {
 public:
  BlockedRange() : _begin(0), _end(0), _grainsize(0) {}
  BlockedRange(int b, int e, int g = 1) : _begin(b), _end(e), _grainsize(g) {}
  int begin() const { return _begin; }
  int end() const { return _end; }
  int grainsize() const { return _grainsize; }

 protected:
  int _begin, _end, _grainsize;
};

template <typename Body>
static inline void parallel_for(const BlockedRange& range, const Body& body) {
  body(range);
}

class Split {};

template <typename Body>
static inline void parallel_reduce(const BlockedRange& range, Body& body) {
  body(range);
}
