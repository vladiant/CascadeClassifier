#include "o_cvboostparams.h"

#include <opencv2/ml/ml.hpp>


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
