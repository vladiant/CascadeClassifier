#ifndef _OPENCV_BOOST_H_
#define _OPENCV_BOOST_H_

#include <opencv2/core/types_c.h>

#include "traincascade_features.h"
#include "utils.h"

// CvCascadeBoostParams <- CvBoostParams <- CvCascadeBoostParams
struct CvCascadeBoostParams : CvBoostParams
{
    float minHitRate;
    float maxFalseAlarm;

    CvCascadeBoostParams();
    CvCascadeBoostParams( int _boostType, float _minHitRate, float _maxFalseAlarm,
                          double _weightTrimRate, int _maxDepth, int _maxWeakCount );
    virtual ~CvCascadeBoostParams() {}
    void write( cv::FileStorage &fs ) const;
    bool read( const cv::FileNode &node );
    virtual void printDefaults() const;
    virtual void printAttrs() const;
    virtual bool scanAttr( const std::string prmName, const std::string val);
};

// CvCascadeBoost <- CvBoost <- CvStatModel
class CvCascadeBoost : public CvBoost
{
public:
    bool train( const CvFeatureEvaluator* _featureEvaluator,
                int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                const CvCascadeBoostParams& _params=CvCascadeBoostParams() );
    float predict( int sampleIdx, bool returnSum = false ) const;

    float getThreshold() const { return threshold; }
    void write( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
    bool read( const cv::FileNode &node, const CvFeatureEvaluator* _featureEvaluator,
               const CvCascadeBoostParams& _params );
    void markUsedFeaturesInMap( cv::Mat& featureMap );
private:
    bool isErrDesired();

    float threshold;
    float minHitRate, maxFalseAlarm;
};

#endif
