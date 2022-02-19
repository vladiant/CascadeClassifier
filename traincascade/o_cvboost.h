#pragma once

#include <opencv2/core/core_c.h>

#include "o_cvboostparams.h"
#include "o_cvstatmodel.h"
#include "o_cvdtreetraindata.h"

// CvCascadeBoost
class CvBoost : public CvStatModel
{
public:
    // Boosting type
    enum { DISCRETE=0, REAL=1, LOGIT=2, GENTLE=3 };

    // Splitting criteria
    enum { DEFAULT=0, GINI=1, MISCLASS=3, SQERR=4 };

    CvBoost();
    virtual ~CvBoost();

    CvBoost( const CvMat* trainData, int tflag,
             const CvMat* responses, const CvMat* varIdx=0,
             const CvMat* sampleIdx=0, const CvMat* varType=0,
             const CvMat* missingDataMask=0,
             CvBoostParams params=CvBoostParams() );

    CV_WRAP virtual void prune( CvSlice slice );

    CV_WRAP virtual void clear();

    CvSeq* get_weak_predictors();

    CvMat* get_weights();
    CvMat* get_subtree_weights();
    CvMat* get_weak_response();
    const CvBoostParams& get_params() const;

protected:

    virtual bool set_params( const CvBoostParams& params );
    virtual void trim_weights();

    CvDTreeTrainData* data;
    CvMat train_data_hdr, responses_hdr;
    cv::Mat train_data_mat, responses_mat;
    CvBoostParams params;
    CvSeq* weak;

    CvMat* active_vars;
    CvMat* active_vars_abs;
    bool have_active_cat_vars;

    CvMat* orig_response;
    CvMat* sum_response;
    CvMat* weak_eval;
    CvMat* subsample_mask;
    CvMat* weights;
    CvMat* subtree_weights;
    bool have_subsample;
};
