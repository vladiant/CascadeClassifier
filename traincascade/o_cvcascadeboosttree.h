#pragma once

#include "o_cvboostree.h"
#include "o_cvdtreenode.h"

// CvCascadeClassifier, CvCascadeBoost
class CvCascadeBoostTree : public CvBoostTree
{
public:
    virtual CvDTreeNode* predict( int sampleIdx ) const;
    void write( cv::FileStorage &fs, const cv::Mat& featureMap );
    void read( const cv::FileNode &node, CvBoost* _ensemble, CvDTreeTrainData* _data );
    void markFeaturesInMap( cv::Mat& featureMap );
protected:
    virtual void split_node_data( CvDTreeNode* n );
};
