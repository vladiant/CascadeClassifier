#include "boost.h"

#include <queue>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ml/ml.hpp>

#include "cascadeclassifier.h"
#include <cmath>

#include "o_cvcascadeboosttree.h"
#include "o_cvcascadeboosttraindata.h"
#include "o_cvdtreenode.h"

using cv::Size;
using cv::Mat;
using cv::Point;
using cv::FileStorage;
using cv::Rect;
using cv::FileNode;


using namespace std;

#define CV_THRESHOLD_EPS (0.00001F)

static inline double
logRatio( double val )
{
    const double eps = 1e-5;

    val = max( val, eps );
    val = min( val, 1. - eps );
    return log( val/(1. - val) );
}


//----------------------------- CascadeBoostParams -------------------------------------------------

CvCascadeBoostParams::CvCascadeBoostParams() : minHitRate( 0.995F), maxFalseAlarm( 0.5F )
{
    boost_type = cv::ml::Boost::GENTLE;
    use_surrogates = use_1se_rule = truncate_pruned_tree = false;
}

CvCascadeBoostParams::CvCascadeBoostParams( int _boostType,
        float _minHitRate, float _maxFalseAlarm,
        double _weightTrimRate, int _maxDepth, int _maxWeakCount ) :
    CvBoostParams( _boostType, _maxWeakCount, _weightTrimRate, _maxDepth, false, nullptr )
{
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

void CvCascadeBoost::update_weights( CvBoostTree* tree )
{
    int n = data->sample_count;
    double sumW = 0.;
    int step = 0;
    float* fdata = nullptr;
    int *sampleIdxBuf = nullptr;
    const int* sampleIdx = nullptr;
    int inn_buf_size = ((params.boost_type == LOGIT) || (params.boost_type == GENTLE) ? n*sizeof(int) : 0) +
                       ( !tree ? n*sizeof(int) : 0 );
    cv::AutoBuffer<uchar> inn_buf(inn_buf_size);
    uchar* cur_inn_buf_pos = inn_buf.data();
    if ( (params.boost_type == LOGIT) || (params.boost_type == GENTLE) )
    {
        step = CV_IS_MAT_CONT(data->responses_copy->type) ?
            1 : data->responses_copy->step / CV_ELEM_SIZE(data->responses_copy->type);
        fdata = data->responses_copy->data.fl;
        sampleIdxBuf = (int*)cur_inn_buf_pos; cur_inn_buf_pos = (uchar*)(sampleIdxBuf + n);
        sampleIdx = data->get_sample_indices( data->data_root, sampleIdxBuf );
    }
    CvMat* buf = data->buf;
    size_t length_buf_row = data->get_length_subbuf();
    if( !tree ) // before training the first tree, initialize weights and other parameters
    {
        int* classLabelsBuf = (int*)cur_inn_buf_pos; cur_inn_buf_pos = (uchar*)(classLabelsBuf + n);
        const int* classLabels = data->get_class_labels(data->data_root, classLabelsBuf);
        // in case of logitboost and gentle adaboost each weak tree is a regression tree,
        // so we need to convert class labels to floating-point values
        double w0 = 1./n;
        double p[2] = { 1, 1 };

        cvReleaseMat( &orig_response );
        cvReleaseMat( &sum_response );
        cvReleaseMat( &weak_eval );
        cvReleaseMat( &subsample_mask );
        cvReleaseMat( &weights );

        orig_response = cvCreateMat( 1, n, CV_32S );
        weak_eval = cvCreateMat( 1, n, CV_64F );
        subsample_mask = cvCreateMat( 1, n, CV_8U );
        weights = cvCreateMat( 1, n, CV_64F );
        subtree_weights = cvCreateMat( 1, n + 2, CV_64F );

        if (data->is_buf_16u)
        {
            auto* labels = (unsigned short*)(buf->data.s + data->data_root->buf_idx*length_buf_row +
                data->data_root->offset + (size_t)(data->work_var_count-1)*data->sample_count);
            for( int i = 0; i < n; i++ )
            {
                // save original categorical responses {0,1}, convert them to {-1,1}
                orig_response->data.i[i] = classLabels[i]*2 - 1;
                // make all the samples active at start.
                // later, in trim_weights() deactivate/reactive again some, if need
                subsample_mask->data.ptr[i] = (uchar)1;
                // make all the initial weights the same.
                weights->data.db[i] = w0*p[classLabels[i]];
                // set the labels to find (from within weak tree learning proc)
                // the particular sample weight, and where to store the response.
                labels[i] = (unsigned short)i;
            }
        }
        else
        {
            int* labels = buf->data.i + data->data_root->buf_idx*length_buf_row +
                data->data_root->offset + (size_t)(data->work_var_count-1)*data->sample_count;

            for( int i = 0; i < n; i++ )
            {
                // save original categorical responses {0,1}, convert them to {-1,1}
                orig_response->data.i[i] = classLabels[i]*2 - 1;
                subsample_mask->data.ptr[i] = (uchar)1;
                weights->data.db[i] = w0*p[classLabels[i]];
                labels[i] = i;
            }
        }

        if( params.boost_type == LOGIT )
        {
            sum_response = cvCreateMat( 1, n, CV_64F );

            for( int i = 0; i < n; i++ )
            {
                sum_response->data.db[i] = 0;
                fdata[sampleIdx[i]*step] = orig_response->data.i[i] > 0 ? 2.f : -2.f;
            }

            // in case of logitboost each weak tree is a regression tree.
            // the target function values are recalculated for each of the trees
            data->is_classifier = false;
        }
        else if( params.boost_type == GENTLE )
        {
            for( int i = 0; i < n; i++ )
                fdata[sampleIdx[i]*step] = (float)orig_response->data.i[i];

            data->is_classifier = false;
        }
    }
    else
    {
        // at this moment, for all the samples that participated in the training of the most
        // recent weak classifier we know the responses. For other samples we need to compute them
        if( have_subsample )
        {
            // invert the subsample mask
            cvXorS( subsample_mask, cvScalar(1.), subsample_mask );

            // run tree through all the non-processed samples
            for( int i = 0; i < n; i++ )
                if( subsample_mask->data.ptr[i] )
                {
                    weak_eval->data.db[i] = (dynamic_cast<CvCascadeBoostTree*>(tree))->predict( i )->value;
                }
        }

        // now update weights and other parameters for each type of boosting
        if( params.boost_type == DISCRETE )
        {
            // Discrete AdaBoost:
            //   weak_eval[i] (=f(x_i)) is in {-1,1}
            //   err = sum(w_i*(f(x_i) != y_i))/sum(w_i)
            //   C = log((1-err)/err)
            //   w_i *= exp(C*(f(x_i) != y_i))

            double C = NAN, err = 0.;
            double scale[] = { 1., 0. };

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i];
                sumW += w;
                err += w*(weak_eval->data.db[i] != orig_response->data.i[i]);
            }

            if( sumW != 0 )
                err /= sumW;
            C = err = -logRatio( err );
            scale[1] = exp(err);

            sumW = 0;
            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*
                    scale[weak_eval->data.db[i] != orig_response->data.i[i]];
                sumW += w;
                weights->data.db[i] = w;
            }

            tree->scale( C );
        }
        else if( params.boost_type == REAL )
        {
            // Real AdaBoost:
            //   weak_eval[i] = f(x_i) = 0.5*log(p(x_i)/(1-p(x_i))), p(x_i)=P(y=1|x_i)
            //   w_i *= exp(-y_i*f(x_i))

            for( int i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i]*weak_eval->data.db[i];
                sumW += w;
                weights->data.db[i] = w;
            }
        }
        else if( params.boost_type == LOGIT )
        {
            // LogitBoost:
            //   weak_eval[i] = f(x_i) in [-z_max,z_max]
            //   sum_response = F(x_i).
            //   F(x_i) += 0.5*f(x_i)
            //   p(x_i) = exp(F(x_i))/(exp(F(x_i)) + exp(-F(x_i))=1/(1+exp(-2*F(x_i)))
            //   reuse weak_eval: weak_eval[i] <- p(x_i)
            //   w_i = p(x_i)*1(1 - p(x_i))
            //   z_i = ((y_i+1)/2 - p(x_i))/(p(x_i)*(1 - p(x_i)))
            //   store z_i to the data->data_root as the new target responses

            const double lbWeightThresh = FLT_EPSILON;
            const double lbZMax = 10.;

            for( int i = 0; i < n; i++ )
            {
                double s = sum_response->data.db[i] + 0.5*weak_eval->data.db[i];
                sum_response->data.db[i] = s;
                weak_eval->data.db[i] = -2*s;
            }

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double p = 1./(1. + weak_eval->data.db[i]);
                double w = p*(1 - p), z = NAN;
                w = MAX( w, lbWeightThresh );
                weights->data.db[i] = w;
                sumW += w;
                if( orig_response->data.i[i] > 0 )
                {
                    z = 1./p;
                    fdata[sampleIdx[i]*step] = (float)min(z, lbZMax);
                }
                else
                {
                    z = 1./(1-p);
                    fdata[sampleIdx[i]*step] = (float)-min(z, lbZMax);
                }
            }
        }
        else
        {
            // Gentle AdaBoost:
            //   weak_eval[i] = f(x_i) in [-1,1]
            //   w_i *= exp(-y_i*f(x_i))
            assert( params.boost_type == GENTLE );

            for( int i = 0; i < n; i++ )
                weak_eval->data.db[i] *= -orig_response->data.i[i];

            cvExp( weak_eval, weak_eval );

            for( int i = 0; i < n; i++ )
            {
                double w = weights->data.db[i] * weak_eval->data.db[i];
                weights->data.db[i] = w;
                sumW += w;
            }
        }
    }

    // renormalize weights
    if( sumW > FLT_EPSILON )
    {
        sumW = 1./sumW;
        for( int i = 0; i < n; ++i )
            weights->data.db[i] *= sumW;
    }
}

bool CvCascadeBoost::train( const CvFeatureEvaluator* _featureEvaluator,
                           int _numSamples,
                           int _precalcValBufSize, int _precalcIdxBufSize,
                           const CvCascadeBoostParams& _params )
{
    bool isTrained = false;
    CV_Assert( !data );
    clear();
    data = new CvCascadeBoostTrainData( _featureEvaluator, _numSamples,
                                        _precalcValBufSize, _precalcIdxBufSize, _params );
    CvMemStorage *storage = cvCreateMemStorage();
    weak = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvBoostTree*), storage );
    storage = nullptr;

    set_params( _params );
    if ( (_params.boost_type == LOGIT) || (_params.boost_type == GENTLE) )
        data->do_responses_copy();

    update_weights( nullptr );

    cout << "+----+---------+---------+" << endl;
    cout << "|  N |    HR   |    FA   |" << endl;
    cout << "+----+---------+---------+" << endl;

    do
    {
        auto* tree = new CvCascadeBoostTree;
        if( !tree->train( data, subsample_mask, this ) )
        {
            delete tree;
            break;
        }
        cvSeqPush( weak, &tree );
        update_weights( tree );
        trim_weights();
        if( cvCountNonZero(subsample_mask) == 0 )
            break;
    }
    while( !isErrDesired() && (weak->total < params.weak_count) );

    if(weak->total > 0)
    {
        data->is_classifier = true;
        data->free_train_data();
        isTrained = true;
    }
    else
        clear();

    return isTrained;
}

float CvCascadeBoost::predict( int sampleIdx, bool returnSum ) const
{
    CV_Assert( weak );
    double sum = 0;
    CvSeqReader reader;
    cvStartReadSeq( weak, &reader );
    cvSetSeqReaderPos( &reader, 0 );
    for( int i = 0; i < weak->total; i++ )
    {
        CvBoostTree* wtree = nullptr;
        CV_READ_SEQ_ELEM( wtree, reader );
        sum += (dynamic_cast<CvCascadeBoostTree*>(wtree))->predict(sampleIdx)->value;
    }
    if( !returnSum )
        sum = sum < threshold - CV_THRESHOLD_EPS ? 0.0 : 1.0;
    return (float)sum;
}

bool CvCascadeBoost::isErrDesired()
{
    int sCount = data->sample_count,
        numPos = 0, numNeg = 0, numFalse = 0, numPosTrue = 0;
    vector<float> eval(sCount);

    for( int i = 0; i < sCount; i++ )
        if( (dynamic_cast<CvCascadeBoostTrainData*>(data))->featureEvaluator->getCls( i ) == 1.0F )
            eval[numPos++] = predict( i, true );

    std::sort(&eval[0], &eval[0] + numPos);

    int thresholdIdx = (int)((1.0F - minHitRate) * numPos);

    threshold = eval[ thresholdIdx ];
    numPosTrue = numPos - thresholdIdx;
    for( int i = thresholdIdx - 1; i >= 0; i--)
        if ( abs( eval[i] - threshold) < FLT_EPSILON )
            numPosTrue++;
    float hitRate = ((float) numPosTrue) / ((float) numPos);

    for( int i = 0; i < sCount; i++ )
    {
        if( (dynamic_cast<CvCascadeBoostTrainData*>(data))->featureEvaluator->getCls( i ) == 0.0F )
        {
            numNeg++;
            if( predict( i ) )
                numFalse++;
        }
    }
    float falseAlarm = ((float) numFalse) / ((float) numNeg);

    cout << "|"; cout.width(4); cout << right << weak->total;
    cout << "|"; cout.width(9); cout << right << hitRate;
    cout << "|"; cout.width(9); cout << right << falseAlarm;
    cout << "|" << endl;
    cout << "+----+---------+---------+" << endl;

    return falseAlarm <= maxFalseAlarm;
}

void CvCascadeBoost::write( FileStorage &fs, const Mat& featureMap ) const
{
    CvCascadeBoostTree* weakTree = nullptr;
    fs << CC_WEAK_COUNT << weak->total;
    fs << CC_STAGE_THRESHOLD << threshold;
    fs << CC_WEAK_CLASSIFIERS << "[";
    for( int wi = 0; wi < weak->total; wi++)
    {
        weakTree = *((CvCascadeBoostTree**) cvGetSeqElem( weak, wi ));
        weakTree->write( fs, featureMap );
    }
    fs << "]";
}

bool CvCascadeBoost::set_params( const CvBoostParams& _params )
{
    minHitRate = ((CvCascadeBoostParams&)_params).minHitRate;
    maxFalseAlarm = ((CvCascadeBoostParams&)_params).maxFalseAlarm;
    return ( ( minHitRate > 0 ) && ( minHitRate < 1) &&
        ( maxFalseAlarm > 0 ) && ( maxFalseAlarm < 1) &&
        CvBoost::set_params( _params ));
}

bool CvCascadeBoost::read( const FileNode &node,
                           const CvFeatureEvaluator* _featureEvaluator,
                           const CvCascadeBoostParams& _params )
{
    CvMemStorage* storage = nullptr;
    clear();
    data = new CvCascadeBoostTrainData( _featureEvaluator, _params );
    set_params( _params );

    node[CC_STAGE_THRESHOLD] >> threshold;
    FileNode rnode = node[CC_WEAK_CLASSIFIERS];

    storage = cvCreateMemStorage();
    weak = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvBoostTree*), storage );
    for(auto && it : rnode)
    {
        auto* tree = new CvCascadeBoostTree();
        tree->read( it, this, data );
        cvSeqPush( weak, &tree );
    }
    return true;
}

void CvCascadeBoost::markUsedFeaturesInMap( Mat& featureMap )
{
    for( int wi = 0; wi < weak->total; wi++ )
    {
        CvCascadeBoostTree* weakTree = *((CvCascadeBoostTree**) cvGetSeqElem( weak, wi ));
        weakTree->markFeaturesInMap( featureMap );
    }
}
