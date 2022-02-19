#pragma once

// CvBoostTree, CvCascadeBoostTrainData, CvCascadeBoostTree, CvDTree, CvDTreeTrainData
struct CvDTreeSplit
{
    int var_idx;
    int condensed_idx;
    int inversed;
    float quality;
    CvDTreeSplit* next;
    union
    {
        int subset[2];
        struct
        {
            float c;
            int split_point;
        }
        ord;
    };
};
