#pragma once

#include <opencv2/core/core_c.h>

// CvCascadeBoostParams
struct CvDTreeParams
{
    int   max_categories;
    int   max_depth;
    int   min_sample_count;
    int   cv_folds;
    bool  use_surrogates;
    bool  use_1se_rule;
    bool  truncate_pruned_tree;
    float regression_accuracy;
    const float* priors;

    CvDTreeParams();
    CvDTreeParams( int max_depth, int min_sample_count,
                   float regression_accuracy, bool use_surrogates,
                   int max_categories, int cv_folds,
                   bool use_1se_rule, bool truncate_pruned_tree,
                   const float* priors );
};


// CvCascadeBoostParams
struct CvBoostParams : public CvDTreeParams
{
    int boost_type;
    int weak_count;
    int split_criteria;
    double weight_trim_rate;

    CvBoostParams();
    CvBoostParams( int boost_type, int weak_count, double weight_trim_rate,
                   int max_depth, bool use_surrogates, const float* priors );
};

// CvCascadeBoost
class CvStatModel
{
public:
    CvStatModel();
    virtual ~CvStatModel();

    virtual void clear();

    // CV_WRAP virtual void save( const char* filename, const char* name=0 ) const;
    // CV_WRAP virtual void load( const char* filename, const char* name=0 );

    // virtual void write( cv::FileStorage& storage, const char* name ) const;
    // virtual void read( const cv::FileNode& node );

protected:
    const char* default_model_name;
};


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

    // virtual bool train( const CvMat* trainData, int tflag,
    //          const CvMat* responses, const CvMat* varIdx=0,
    //          const CvMat* sampleIdx=0, const CvMat* varType=0,
    //          const CvMat* missingDataMask=0,
    //          CvBoostParams params=CvBoostParams(),
    //          bool update=false );

    // virtual bool train( CvMLData* data,
    //          CvBoostParams params=CvBoostParams(),
    //          bool update=false );

    // virtual float predict( const CvMat* sample, const CvMat* missing=0,
    //                        CvMat* weak_responses=0, CvSlice slice=CV_WHOLE_SEQ,
    //                        bool raw_mode=false, bool return_sum=false ) const;

    // CvBoost( const cv::Mat& trainData, int tflag,
    //         const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
    //         const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(),
    //         const cv::Mat& missingDataMask=cv::Mat(),
    //         CvBoostParams params=CvBoostParams() );

    // CV_WRAP virtual bool train( const cv::Mat& trainData, int tflag,
    //                    const cv::Mat& responses, const cv::Mat& varIdx=cv::Mat(),
    //                    const cv::Mat& sampleIdx=cv::Mat(), const cv::Mat& varType=cv::Mat(),
    //                    const cv::Mat& missingDataMask=cv::Mat(),
    //                    CvBoostParams params=CvBoostParams(),
    //                    bool update=false );

    // CV_WRAP virtual float predict( const cv::Mat& sample, const cv::Mat& missing=cv::Mat(),
    //                                const cv::Range& slice=cv::Range::all(), bool rawMode=false,
    //                                bool returnSum=false ) const;

    // virtual float calc_error( CvMLData* _data, int type , std::vector<float> *resp = 0 ); // type in {CV_TRAIN_ERROR, CV_TEST_ERROR}

    CV_WRAP virtual void prune( CvSlice slice );

    CV_WRAP virtual void clear();

    // virtual void write( CvFileStorage* storage, const char* name ) const;
    // virtual void read( CvFileStorage* storage, CvFileNode* node );
    // virtual const CvMat* get_active_vars(bool absolute_idx=true);

    CvSeq* get_weak_predictors();

    // CvMat* get_weights();
    // CvMat* get_subtree_weights();
    // CvMat* get_weak_response();
    // const CvBoostParams& get_params() const;
    // const CvDTreeTrainData* get_data() const;

protected:

    // virtual bool set_params( const CvBoostParams& params );
    // virtual void update_weights( CvBoostTree* tree );
    // virtual void trim_weights();
    // virtual void write_params( CvFileStorage* fs ) const;
    // virtual void read_params( CvFileStorage* fs, CvFileNode* node );

    // virtual void initialize_weights(double (&p)[2]);

    // CvDTreeTrainData* data;
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
