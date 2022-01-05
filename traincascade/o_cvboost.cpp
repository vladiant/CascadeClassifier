#include "o_cvboost.h"

#include <opencv2/ml/ml.hpp>

#include "o_cvdtreeparams.h"
#include "o_cvboostree.h"


CvBoost::CvBoost()
{
    data = nullptr;
    weak = nullptr;
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
        for( i = 0; i < count; i++ )
        {
            CvBoostTree* w;
            CV_READ_SEQ_ELEM( w, reader );
            delete w;
        }

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
    if( data )
        delete data;
    weak = nullptr;
    data = nullptr;
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

CvMat*
CvBoost::get_weak_response()
{
    return weak_eval;
}

CvMat*
CvBoost::get_subtree_weights()
{
    return subtree_weights;
}

const CvBoostParams&
CvBoost::get_params() const
{
    return params;
}

CvMat*
CvBoost::get_weights()
{
    return weights;
}
