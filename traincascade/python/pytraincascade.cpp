#include <stdexcept>
#include <string>

#include <opencv2/core/utility.hpp>
#include <opencv2/ml.hpp>

#include <pybind11/pybind11.h>

#include "boost.h"
#include "cascadeclassifier.h"
#include "haarfeatures.h"

namespace py = pybind11;

enum class FeatureType
{
    HAAR = CvFeatureParams::HAAR,
    LBP = CvFeatureParams::LBP,
    HOG = CvFeatureParams::HOG
};

enum class HaarMode
{
    BASIC = CvHaarFeatureParams::BASIC,
    CORE = CvHaarFeatureParams::CORE,
    ALL = CvHaarFeatureParams::ALL
};

enum class BoostType
{
    DISCRETE = cv::ml::Boost::DISCRETE,
    REAL = cv::ml::Boost::REAL,
    LOGIT = cv::ml::Boost::LOGIT,
    GENTLE = cv::ml::Boost::GENTLE
};

struct TrainerOptions
{
    TrainerOptions()
    {
        const CvCascadeParams cascadeDefaults;
        const CvCascadeBoostParams boostDefaults;
        const CvHaarFeatureParams haarDefaults;

        num_pos = 2000;
        num_neg = 1000;
        num_stages = 20;
        num_threads = cv::getNumThreads();
        precalc_val_buf_size_mb = 1024;
        precalc_idx_buf_size_mb = 1024;
        base_format_save = false;
        acceptance_ratio_break_value = -1.0;

        stage_type = cascadeDefaults.stageType;
        feature_type = static_cast<FeatureType>(cascadeDefaults.featureType);
        win_width = cascadeDefaults.winSize.width;
        win_height = cascadeDefaults.winSize.height;

        boost_type = static_cast<BoostType>(boostDefaults.boost_type);
        min_hit_rate = boostDefaults.minHitRate;
        max_false_alarm_rate = boostDefaults.maxFalseAlarm;
        weight_trim_rate = boostDefaults.weight_trim_rate;
        max_depth = boostDefaults.max_depth;
        max_weak_count = boostDefaults.weak_count;

        haar_mode = static_cast<HaarMode>(haarDefaults.mode);
    }

    std::string data_dir;
    std::string vec_file;
    std::string bg_file;

    int num_pos;
    int num_neg;
    int num_stages;
    int num_threads;
    int precalc_val_buf_size_mb;
    int precalc_idx_buf_size_mb;
    bool base_format_save;
    double acceptance_ratio_break_value;

    int stage_type;
    FeatureType feature_type;
    int win_width;
    int win_height;

    BoostType boost_type;
    float min_hit_rate;
    float max_false_alarm_rate;
    double weight_trim_rate;
    int max_depth;
    int max_weak_count;

    HaarMode haar_mode;
};

static bool train_with_options(const TrainerOptions& options)
{
    if (options.data_dir.empty())
    {
        throw std::invalid_argument("options.data_dir must not be empty");
    }
    if (options.vec_file.empty())
    {
        throw std::invalid_argument("options.vec_file must not be empty");
    }
    if (options.bg_file.empty())
    {
        throw std::invalid_argument("options.bg_file must not be empty");
    }

    CvCascadeParams cascade_params;
    cascade_params.stageType = options.stage_type;
    cascade_params.featureType = static_cast<int>(options.feature_type);
    cascade_params.winSize = cv::Size(options.win_width, options.win_height);

    CvCascadeBoostParams stage_params;
    stage_params.boost_type = static_cast<int>(options.boost_type);
    stage_params.minHitRate = options.min_hit_rate;
    stage_params.maxFalseAlarm = options.max_false_alarm_rate;
    stage_params.weight_trim_rate = options.weight_trim_rate;
    stage_params.max_depth = options.max_depth;
    stage_params.weak_count = options.max_weak_count;

    cv::Ptr<CvFeatureParams> feature_params = CvFeatureParams::create(cascade_params.featureType);
    if (feature_params.empty())
    {
        throw std::invalid_argument("unsupported feature_type");
    }

    if (cascade_params.featureType == CvFeatureParams::HAAR)
    {
        auto* haar_params = dynamic_cast<CvHaarFeatureParams*>(feature_params.get());
        if (haar_params == nullptr)
        {
            throw std::runtime_error("failed to initialize HAAR feature parameters");
        }
        haar_params->mode = static_cast<int>(options.haar_mode);
    }

    cv::setNumThreads(options.num_threads);

    CvCascadeClassifier classifier;
    return classifier.train(
        options.data_dir,
        options.vec_file,
        options.bg_file,
        options.num_pos,
        options.num_neg,
        options.precalc_val_buf_size_mb,
        options.precalc_idx_buf_size_mb,
        options.num_stages,
        cascade_params,
        *feature_params,
        stage_params,
        options.base_format_save,
        options.acceptance_ratio_break_value
    );
}

PYBIND11_MODULE(pytraincascade, module)
{
    module.doc() = "Python bindings for the cascade trainer";

    py::enum_<FeatureType>(module, "FeatureType")
        .value("HAAR", FeatureType::HAAR)
        .value("LBP", FeatureType::LBP)
        .value("HOG", FeatureType::HOG)
        .export_values();

    py::enum_<HaarMode>(module, "HaarMode")
        .value("BASIC", HaarMode::BASIC)
        .value("CORE", HaarMode::CORE)
        .value("ALL", HaarMode::ALL)
        .export_values();

    py::enum_<BoostType>(module, "BoostType")
        .value("DISCRETE", BoostType::DISCRETE)
        .value("REAL", BoostType::REAL)
        .value("LOGIT", BoostType::LOGIT)
        .value("GENTLE", BoostType::GENTLE)
        .export_values();

    py::class_<TrainerOptions>(module, "TrainerOptions")
        .def(py::init<>())
        .def_readwrite("data_dir", &TrainerOptions::data_dir)
        .def_readwrite("vec_file", &TrainerOptions::vec_file)
        .def_readwrite("bg_file", &TrainerOptions::bg_file)
        .def_readwrite("num_pos", &TrainerOptions::num_pos)
        .def_readwrite("num_neg", &TrainerOptions::num_neg)
        .def_readwrite("num_stages", &TrainerOptions::num_stages)
        .def_readwrite("num_threads", &TrainerOptions::num_threads)
        .def_readwrite("precalc_val_buf_size_mb", &TrainerOptions::precalc_val_buf_size_mb)
        .def_readwrite("precalc_idx_buf_size_mb", &TrainerOptions::precalc_idx_buf_size_mb)
        .def_readwrite("base_format_save", &TrainerOptions::base_format_save)
        .def_readwrite("acceptance_ratio_break_value", &TrainerOptions::acceptance_ratio_break_value)
        .def_readwrite("stage_type", &TrainerOptions::stage_type)
        .def_readwrite("feature_type", &TrainerOptions::feature_type)
        .def_readwrite("win_width", &TrainerOptions::win_width)
        .def_readwrite("win_height", &TrainerOptions::win_height)
        .def_readwrite("boost_type", &TrainerOptions::boost_type)
        .def_readwrite("min_hit_rate", &TrainerOptions::min_hit_rate)
        .def_readwrite("max_false_alarm_rate", &TrainerOptions::max_false_alarm_rate)
        .def_readwrite("weight_trim_rate", &TrainerOptions::weight_trim_rate)
        .def_readwrite("max_depth", &TrainerOptions::max_depth)
        .def_readwrite("max_weak_count", &TrainerOptions::max_weak_count)
        .def_readwrite("haar_mode", &TrainerOptions::haar_mode);

    module.def(
        "train",
        [](const TrainerOptions& options)
        {
            try
            {
                return train_with_options(options);
            }
            catch (const cv::Exception& error)
            {
                throw std::runtime_error(error.what());
            }
        },
        py::arg("options"),
        "Train a cascade classifier with the provided options."
    );
}
