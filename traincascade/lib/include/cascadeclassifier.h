/**
 * @file cascadeclassifier.h
 * @brief Top-level cascade classifier training driver.
 *
 * Defines @ref CvCascadeParams (per-cascade settings: feature type, window
 * size, stage type) and @ref CvCascadeClassifier, the orchestrator that
 * drives the multi-stage training loop. The class iterates over stages,
 * trains a @ref CvCascadeBoost per stage, samples positives/negatives
 * through @ref CvCascadeImageReader and writes the resulting cascade
 * (along with @c params.xml and @c cascade.xml) to disk.
 *
 * String constants prefixed @c CC_ are XML node names used by the cascade
 * file format and shared with OpenCV's runtime detector loader.
 */

#ifndef _OPENCV_CASCADECLASSIFIER_H_
#define _OPENCV_CASCADECLASSIFIER_H_

#include <ctime>
#include "traincascade_features.h"
#include "haarfeatures.h"
#include "lbpfeatures.h"
#include "HOGfeatures.h" //new
#include "boost.h"

/// Default file name of the trained cascade written into the output directory.
#define CC_CASCADE_FILENAME "cascade.xml"
/// File name used to persist the training parameter set, allowing a session to be resumed.
#define CC_PARAMS_FILENAME "params.xml"

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE "stageType"
#define CC_FEATURE_TYPE "featureType"
#define CC_HEIGHT "height"
#define CC_WIDTH  "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_BOOST_TYPE       "boostType"
#define CC_DISCRETE_BOOST   "DAB"
#define CC_REAL_BOOST       "RAB"
#define CC_LOGIT_BOOST      "LB"
#define CC_GENTLE_BOOST     "GAB"
#define CC_MINHITRATE       "minHitRate"
#define CC_MAXFALSEALARM    "maxFalseAlarm"
#define CC_TRIM_RATE        "weightTrimRate"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       FEATURES
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"
#define CC_FEATURE_SIZE   "featSize"

#define CC_HAAR        "HAAR"
#define CC_MODE        "mode"
#define CC_MODE_BASIC  "BASIC"
#define CC_MODE_CORE   "CORE"
#define CC_MODE_ALL    "ALL"
#define CC_RECTS       "rects"
#define CC_TILTED      "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CC_HOG "HOG"

/// Cross-platform wall-clock timer macro used by the training driver to
/// stamp elapsed-time messages. Resolves to @c clock() on Windows and
/// @c time() elsewhere so the trainer compiles without POSIX-only headers.
#ifdef _WIN32
#define TIME( arg ) (((double) clock()) / CLOCKS_PER_SEC)
#else
#define TIME( arg ) (time( arg ))
#endif

/**
 * @brief Top-level parameters describing the cascade as a whole.
 *
 * Captures the high-level choices that apply to every stage: the boosting
 * variant used at each stage (@c stageType), the feature family
 * (@c featureType, one of HAAR/LBP/HOG via @ref CvFeatureParams) and the
 * detection window size (@c winSize). Values are persisted to and loaded
 * from the @c params.xml / @c cascade.xml file alongside per-stage data.
 */
class CvCascadeParams : public CvParams
{
public:
    enum { BOOST = 0 }; ///< Supported stage types (currently only boosting).
    static const int defaultStageType = BOOST;
    static const int defaultFeatureType = CvFeatureParams::HAAR;

    CvCascadeParams();
    CvCascadeParams( int _stageType, int _featureType );
    /// Serialize the cascade-level parameters to an open @c FileStorage node.
    void write( cv::FileStorage &fs ) const override;
    /// Restore parameters from an XML/YAML node; returns @c false on malformed input.
    bool read( const cv::FileNode &node ) override;

    void printDefaults() const override;
    void printAttrs() const override;
    /// Parse a single command-line attribute (e.g. @c -featureType, @c -w, @c -h).
    bool scanAttr( const std::string prmName, const std::string val ) override;

    int stageType;     ///< Which stage classifier flavor to train (currently always BOOST).
    int featureType;   ///< Feature family: HAAR, LBP or HOG (see @ref CvFeatureParams).
    cv::Size winSize;  ///< Sub-window size used to crop positives and scan negatives.
};

/**
 * @brief Driver class that trains a multi-stage Haar/LBP/HOG cascade.
 *
 * The cascade-training loop is implemented in @ref train: for each stage
 * the trainer (1) refreshes the working sample set with positives still
 * accepted by the previous stages and freshly mined negatives, (2) trains
 * a single @ref CvCascadeBoost so its true-positive rate meets
 * @c minHitRate while staying below @c maxFalseAlarm, and (3) appends the
 * resulting weak ensemble to @c stageClassifiers. Intermediate state is
 * checkpointed under @c cascadeDirName so interrupted training can resume.
 *
 * After all @c numStages have been completed the cascade is exported via
 * @ref save in either the modern @c cascade.xml format or the legacy
 * "baseFormatSave" layout produced by older OpenCV versions.
 */
class CvCascadeClassifier
{
public:
    /**
     * @brief Train a complete cascade and write it to disk.
     *
     * @param _cascadeDirName Output directory; receives intermediate XML
     *        files plus the final @c cascade.xml.
     * @param _posFilename Path to the @c .vec file produced by
     *        opencv_createsamples that holds positive samples.
     * @param _negFilename Path to a text file listing background images.
     * @param _numPos Number of positives to consume per stage.
     * @param _numNeg Number of false positives to mine per stage.
     * @param _precalcValBufSize Buffer size (MB) for precomputed feature values.
     * @param _precalcIdxBufSize Buffer size (MB) for precomputed sorted indices.
     * @param _numStages Maximum number of cascade stages to train.
     * @param _cascadeParams Cascade-wide parameters (window size, feature type).
     * @param _featureParams Feature-family-specific parameters.
     * @param _stageParams Boosting parameters used by every stage.
     * @param baseFormatSave When @c true also writes the legacy XML layout.
     * @param acceptanceRatioBreakValue Stop training early if the running
     *        acceptance ratio falls below this threshold; @c -1 disables it.
     * @return @c true on successful training, @c false if the trainer could
     *         not collect enough positive/negative samples for some stage.
     */
    bool train( const std::string& _cascadeDirName,
                const std::string& _posFilename,
                const std::string& _negFilename,
                int _numPos, int _numNeg,
                int _precalcValBufSize, int _precalcIdxBufSize,
                int _numStages,
                const CvCascadeParams& _cascadeParams,
                const CvFeatureParams& _featureParams,
                const CvCascadeBoostParams& _stageParams,
                bool baseFormatSave = false,
                double acceptanceRatioBreakValue = -1.0 );
private:
    /// Run @p sampleIdx through every already-trained stage and report
    /// whether all stages accepted it (1) or some rejected it (0).
    int predict( int sampleIdx );
    /// Persist the cascade to @p cascadeDirName; @p baseFormat selects the legacy XML layout.
    void save( const std::string& cascadeDirName, bool baseFormat = false );
    /// Resume an interrupted training session from @c params.xml / stage XMLs.
    bool load( const std::string& cascadeDirName );
    /// Refill the working sample set for the next stage; rejects negatives
    /// the cascade already discards. @p acceptanceRatio reports how rare
    /// usable negatives have become (used by the early-stop check).
    bool updateTrainingSet( double minimumAcceptanceRatio, double& acceptanceRatio );
    /// Mine a contiguous block of samples that pass the current cascade.
    int fillPassedSamples( int first, int count, bool isPositive, double requiredAcceptanceRatio, int64& consumed );

    void writeParams( cv::FileStorage &fs ) const;
    void writeStages( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
    void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
    bool readParams( const cv::FileNode &node );
    bool readStages( const cv::FileNode &node );

    /// Build a dense remapping from global feature indices to the subset
    /// that any stage actually selected; used to compact the saved cascade.
    void getUsedFeaturesIdxMap( cv::Mat& featureMap );

    CvCascadeParams cascadeParams;                              ///< Cascade-wide parameters.
    cv::Ptr<CvFeatureParams> featureParams;                     ///< Feature-family parameters.
    cv::Ptr<CvCascadeBoostParams> stageParams;                  ///< Boosting parameters shared by all stages.

    cv::Ptr<CvFeatureEvaluator> featureEvaluator;               ///< Computes feature responses on cached images.
    std::vector< cv::Ptr<CvCascadeBoost> > stageClassifiers;    ///< Trained stages in order of evaluation.
    CvCascadeImageReader imgReader;                             ///< Streaming source of positives and negatives.
    int numStages, curNumSamples;
    int numPos, numNeg;
};

#endif
