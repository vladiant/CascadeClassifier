#include "utils.h"

#include <climits>

#include <opencv2/ml/ml.hpp>

CvDTreeParams::CvDTreeParams() : max_categories(10), max_depth(INT_MAX), min_sample_count(10),
    cv_folds(10), use_surrogates(true), use_1se_rule(true),
    truncate_pruned_tree(true), regression_accuracy(0.01f), priors(0)
{}

CvDTreeParams::CvDTreeParams( int _max_depth, int _min_sample_count,
                              float _regression_accuracy, bool _use_surrogates,
                              int _max_categories, int _cv_folds,
                              bool _use_1se_rule, bool _truncate_pruned_tree,
                              const float* _priors ) :
    max_categories(_max_categories), max_depth(_max_depth),
    min_sample_count(_min_sample_count), cv_folds (_cv_folds),
    use_surrogates(_use_surrogates), use_1se_rule(_use_1se_rule),
    truncate_pruned_tree(_truncate_pruned_tree),
    regression_accuracy(_regression_accuracy),
    priors(_priors)
{}

CvBoostParams::CvBoostParams()
{
    boost_type = cv::ml::Boost::REAL;
    weak_count = 100;
    weight_trim_rate = 0.95;
    cv_folds = 0;
    max_depth = 1;
}


CvBoostParams::CvBoostParams( int _boost_type, int _weak_count,
                                        double _weight_trim_rate, int _max_depth,
                                        bool _use_surrogates, const float* _priors )
{
    boost_type = _boost_type;
    weak_count = _weak_count;
    weight_trim_rate = _weight_trim_rate;
    // split_criteria = CvBoost::DEFAULT;
    cv_folds = 0;
    max_depth = _max_depth;
    use_surrogates = _use_surrogates;
    priors = _priors;
}


CvStatModel::CvStatModel()
{
    default_model_name = "my_stat_model";
}


CvStatModel::~CvStatModel()
{
    clear();
}


void CvStatModel::clear()
{
}


CvBoost::CvBoost()
{
    // data = 0;
    weak = 0;
    default_model_name = "my_boost_tree";

    active_vars = active_vars_abs = orig_response = sum_response = weak_eval =
        subsample_mask = weights = subtree_weights = 0;
    have_active_cat_vars = have_subsample = false;

    clear();
}

void CvBoost::prune( CvSlice slice )
{
    if( weak && weak->total > 0 )
    {
        CvSeqReader reader;
        int i, count = cvSliceLength( slice, weak );

        cvStartReadSeq( weak, &reader );
        cvSetSeqReaderPos( &reader, slice.start_index );

        static_cast<void>(i);
        static_cast<void>(count);
        // for( i = 0; i < count; i++ )
        // {
        //     CvBoostTree* w;
        //     CV_READ_SEQ_ELEM( w, reader );
        //     delete w;
        // }

        cvSeqRemoveSlice( weak, slice );
    }
}


void CvBoost::clear()
{
    if( weak )
    {
        prune( CV_WHOLE_SEQ );
        cvReleaseMemStorage( &weak->storage );
    }
    // if( data )
    //     delete data;
    weak = 0;
    // data = 0;
    cvReleaseMat( &active_vars );
    cvReleaseMat( &active_vars_abs );
    cvReleaseMat( &orig_response );
    cvReleaseMat( &sum_response );
    cvReleaseMat( &weak_eval );
    cvReleaseMat( &subsample_mask );
    cvReleaseMat( &weights );
    cvReleaseMat( &subtree_weights );

    have_subsample = false;
}

CvBoost::~CvBoost()
{
    clear();
}

CvSeq* CvBoost::get_weak_predictors()
{
    return weak;
}
