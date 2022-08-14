#pragma once

// CvBoostTree, CvCascadeBoostTrainData, CvCascadeBoostTree, CvDTree,
// CvDTreeTrainData
struct CvDTreeSplit {
  int var_idx;
  int condensed_idx;
  int inversed;
  float quality;
  CvDTreeSplit* next;
  struct Ord {
      float c;
      int split_point;
  };
  union {
    int subset[2];
    Ord ord;
  };
};