#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ml/ml.hpp>

using cv::Size;
using cv::Mat;
using cv::Point;
using cv::FileStorage;
using cv::Rect;
using cv::Ptr;
using cv::FileNode;
using cv::Mat_;
using cv::Range;
using cv::FileNodeIterator;
using cv::ParallelLoopBody;


using cv::Size;
using cv::Mat;
using cv::Point;
using cv::FileStorage;
using cv::Rect;
using cv::Ptr;
using cv::FileNode;
using cv::Mat_;
using cv::Range;
using cv::FileNodeIterator;
using cv::ParallelLoopBody;


#include "boost.h"
#include "cascadeclassifier.h"
#include <queue>

using namespace std;

static inline double
logRatio( double val )
{
    const double eps = 1e-5;

    val = max( val, eps );
    val = min( val, 1. - eps );
    return log( val/(1. - val) );
}

template<typename T, typename Idx>
class LessThanIdx
{
public:
    LessThanIdx( const T* _arr ) : arr(_arr) {}
    bool operator()(Idx a, Idx b) const { return arr[a] < arr[b]; }
    const T* arr;
};

static inline int cvAlign( int size, int align )
{
    CV_DbgAssert( (align & (align-1)) == 0 && size < INT_MAX );
    return (size + align - 1) & -align;
}

#define CV_THRESHOLD_EPS (0.00001F)

static const int MinBlockSize = 1 << 16;
static const int BlockSizeDelta = 1 << 10;

//----------------------------- CascadeBoostParams -------------------------------------------------

CvCascadeBoostParams::CvCascadeBoostParams() : minHitRate( 0.995F), maxFalseAlarm( 0.5F )
{
    // CvDTreeParams
    cv_folds = 10;
    use_surrogates = true;
    use_1se_rule = true;
    truncate_pruned_tree = true;
    regression_accuracy = 0.01f;
    priors = nullptr;

    // CvBoostParams
    boost_type = cv::ml::Boost::REAL;
    weak_count = 100;
    weight_trim_rate = 0.95;
    cv_folds = 0;
    max_depth = 1;

    boost_type = cv::ml::Boost::GENTLE;
    use_surrogates = use_1se_rule = truncate_pruned_tree = false;
}

CvCascadeBoostParams::CvCascadeBoostParams( int _boostType,
        float _minHitRate, float _maxFalseAlarm,
        double _weightTrimRate, int _maxDepth, int _maxWeakCount )
{
    // CvBoostParams
    boost_type = _boostType;
    weak_count = _maxWeakCount;
    weight_trim_rate = _weightTrimRate;
    cv_folds = 0;
    max_depth = _maxDepth;
    use_surrogates = false;
    priors = nullptr;

    boost_type = cv::ml::Boost::GENTLE;
    minHitRate = _minHitRate;
    maxFalseAlarm = _maxFalseAlarm;
    use_surrogates = use_1se_rule = truncate_pruned_tree = false;
}

void CvCascadeBoostParams::write( FileStorage &fs ) const
{
    string boostTypeStr = boost_type == cv::ml::Boost::DISCRETE ? CC_DISCRETE_BOOST :
                          boost_type == cv::ml::Boost::REAL ? CC_REAL_BOOST :
                          boost_type == cv::ml::Boost::LOGIT ? CC_LOGIT_BOOST :
                          boost_type == cv::ml::Boost::GENTLE ? CC_GENTLE_BOOST : string();
    CV_Assert( !boostTypeStr.empty() );
    fs << CC_BOOST_TYPE << boostTypeStr;
    fs << CC_MINHITRATE << minHitRate;
    fs << CC_MAXFALSEALARM << maxFalseAlarm;
    fs << CC_TRIM_RATE << weight_trim_rate;
    fs << CC_MAX_DEPTH << max_depth;
    fs << CC_WEAK_COUNT << weak_count;
}

bool CvCascadeBoostParams::read( const FileNode &node )
{
    string boostTypeStr;
    FileNode rnode = node[CC_BOOST_TYPE];
    rnode >> boostTypeStr;
    boost_type = !boostTypeStr.compare( CC_DISCRETE_BOOST ) ? cv::ml::Boost::DISCRETE :
                 !boostTypeStr.compare( CC_REAL_BOOST ) ? cv::ml::Boost::REAL :
                 !boostTypeStr.compare( CC_LOGIT_BOOST ) ? cv::ml::Boost::LOGIT :
                 !boostTypeStr.compare( CC_GENTLE_BOOST ) ? cv::ml::Boost::GENTLE : -1;
    if (boost_type == -1)
        CV_Error( cv::Error::StsBadArg, "unsupported Boost type" );
    node[CC_MINHITRATE] >> minHitRate;
    node[CC_MAXFALSEALARM] >> maxFalseAlarm;
    node[CC_TRIM_RATE] >> weight_trim_rate ;
    node[CC_MAX_DEPTH] >> max_depth ;
    node[CC_WEAK_COUNT] >> weak_count ;
    if ( minHitRate <= 0 || minHitRate > 1 ||
         maxFalseAlarm <= 0 || maxFalseAlarm > 1 ||
         weight_trim_rate <= 0 || weight_trim_rate > 1 ||
         max_depth <= 0 || weak_count <= 0 )
        CV_Error( cv::Error::StsBadArg, "bad parameters range");
    return true;
}

void CvCascadeBoostParams::printDefaults() const
{
    cout << "--boostParams--" << endl;
    cout << "  [-bt <{" << CC_DISCRETE_BOOST << ", "
                        << CC_REAL_BOOST << ", "
                        << CC_LOGIT_BOOST ", "
                        << CC_GENTLE_BOOST << "(default)}>]" << endl;
    cout << "  [-minHitRate <min_hit_rate> = " << minHitRate << ">]" << endl;
    cout << "  [-maxFalseAlarmRate <max_false_alarm_rate = " << maxFalseAlarm << ">]" << endl;
    cout << "  [-weightTrimRate <weight_trim_rate = " << weight_trim_rate << ">]" << endl;
    cout << "  [-maxDepth <max_depth_of_weak_tree = " << max_depth << ">]" << endl;
    cout << "  [-maxWeakCount <max_weak_tree_count = " << weak_count << ">]" << endl;
}

void CvCascadeBoostParams::printAttrs() const
{
    string boostTypeStr = boost_type == cv::ml::Boost::DISCRETE ? CC_DISCRETE_BOOST :
                          boost_type == cv::ml::Boost::REAL ? CC_REAL_BOOST :
                          boost_type == cv::ml::Boost::LOGIT  ? CC_LOGIT_BOOST :
                          boost_type == cv::ml::Boost::GENTLE ? CC_GENTLE_BOOST : string();
    CV_Assert( !boostTypeStr.empty() );
    cout << "boostType: " << boostTypeStr << endl;
    cout << "minHitRate: " << minHitRate << endl;
    cout << "maxFalseAlarmRate: " <<  maxFalseAlarm << endl;
    cout << "weightTrimRate: " << weight_trim_rate << endl;
    cout << "maxDepth: " << max_depth << endl;
    cout << "maxWeakCount: " << weak_count << endl;
}

bool CvCascadeBoostParams::scanAttr( const string prmName, const string val)
{
    bool res = true;

    if( !prmName.compare( "-bt" ) )
    {
        boost_type = !val.compare( CC_DISCRETE_BOOST ) ? cv::ml::Boost::DISCRETE :
                     !val.compare( CC_REAL_BOOST ) ? cv::ml::Boost::REAL :
                     !val.compare( CC_LOGIT_BOOST ) ? cv::ml::Boost::LOGIT :
                     !val.compare( CC_GENTLE_BOOST ) ? cv::ml::Boost::GENTLE : -1;
        if (boost_type == -1)
            res = false;
    }
    else if( !prmName.compare( "-minHitRate" ) )
    {
        minHitRate = (float) atof( val.c_str() );
    }
    else if( !prmName.compare( "-maxFalseAlarmRate" ) )
    {
        maxFalseAlarm = (float) atof( val.c_str() );
    }
    else if( !prmName.compare( "-weightTrimRate" ) )
    {
        weight_trim_rate = (float) atof( val.c_str() );
    }
    else if( !prmName.compare( "-maxDepth" ) )
    {
        max_depth = atoi( val.c_str() );
    }
    else if( !prmName.compare( "-maxWeakCount" ) )
    {
        weak_count = atoi( val.c_str() );
    }
    else
        res = false;

    return res;
}


//----------------------------------- CascadeBoost --------------------------------------

bool CvCascadeBoost::train( const CvFeatureEvaluator* _featureEvaluator,
                           int _numSamples,
                           int _precalcValBufSize, int _precalcIdxBufSize,
                           const CvCascadeBoostParams& _params )
{
    bool isTrained = false;
    static_cast<void>(_featureEvaluator);
    static_cast<void>(_numSamples);
    static_cast<void>(_precalcValBufSize);
    static_cast<void>(_precalcIdxBufSize);
    static_cast<void>(_params);
    return isTrained;
}

float CvCascadeBoost::predict( int sampleIdx, bool returnSum ) const
{
    double sum = 0;
    static_cast<void>(sampleIdx);
    static_cast<void>(returnSum);
    return (float)sum;
}

bool CvCascadeBoost::isErrDesired()
{
    return false;
}

void CvCascadeBoost::write( FileStorage &fs, const Mat& featureMap ) const
{
    static_cast<void>(fs);
    static_cast<void>(featureMap);
}

bool CvCascadeBoost::read( const FileNode &node,
                           const CvFeatureEvaluator* _featureEvaluator,
                           const CvCascadeBoostParams& _params )
{
    static_cast<void>(node);
    static_cast<void>(_featureEvaluator);
    static_cast<void>(_params);
    return true;
}

void CvCascadeBoost::markUsedFeaturesInMap( Mat& featureMap )
{
    static_cast<void>(featureMap);
}
