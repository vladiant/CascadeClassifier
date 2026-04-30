/**
 * @file boost.h
 * @brief Cascade-stage boosting classifier and its parameter struct.
 *
 * Hosts the two cascade-specific subclasses derived from the legacy
 * @c CvBoost / @c CvBoostParams machinery (see the @c o_cvboost*.h family):
 *  - @ref CvCascadeBoostParams adds the cascade-only knobs @c minHitRate
 *    and @c maxFalseAlarm to the boosting parameter set.
 *  - @ref CvCascadeBoost overrides the boosting training loop so it stops
 *    when those rates are met and exposes the per-stage decision threshold
 *    used at runtime.
 */

#ifndef _OPENCV_BOOST_H_
#define _OPENCV_BOOST_H_

#include <opencv2/core/types_c.h>

#include "o_cvboost.h"
#include "o_cvboostparams.h"
#include "o_cvboostree.h"

#include "traincascade_features.h"


/**
 * @brief Parameters for a single cascade stage trained as a boosted ensemble.
 *
 * Extends @ref CvBoostParams (boost type, weak-count cap, weight-trim
 * threshold, weak-tree max depth) with two cascade-specific targets:
 *  - @c minHitRate — minimum fraction of positives the stage must keep.
 *  - @c maxFalseAlarm — maximum fraction of negatives allowed through.
 *
 * Training stops as soon as both rates are satisfied, even if @c weak_count
 * weak learners have not been added yet.
 */
struct CvCascadeBoostParams : CvBoostParams
{
    float minHitRate;     ///< Lower bound on the per-stage true-positive rate.
    float maxFalseAlarm;  ///< Upper bound on the per-stage false-positive rate.

    CvCascadeBoostParams();
    CvCascadeBoostParams( int _boostType, float _minHitRate, float _maxFalseAlarm,
                          double _weightTrimRate, int _maxDepth, int _maxWeakCount );
    virtual ~CvCascadeBoostParams() {}
    /// Persist parameters to an XML/YAML node (used by @c params.xml).
    void write( cv::FileStorage &fs ) const;
    /// Restore parameters from a node; returns @c false on malformed input.
    bool read( const cv::FileNode &node );
    virtual void printDefaults() const;
    virtual void printAttrs() const;
    /// Parse one @c -name value command-line attribute.
    virtual bool scanAttr( const std::string prmName, const std::string val);
};

/**
 * @brief Boosted ensemble representing a single cascade stage.
 *
 * Built on top of OpenCV's legacy @c CvBoost; the cascade trainer adds
 * weak learners until either @c minHitRate / @c maxFalseAlarm are reached
 * or the configured weak-count cap is hit. After training the stage stores
 * a real-valued decision @c threshold tuned so the desired hit rate is met
 * on the working positive set.
 *
 * At runtime the stage accepts a sample iff the sum of weak responses
 * exceeds @c threshold.
 */
class CvCascadeBoost : public CvBoost
{
public:
    /**
     * @brief Train one cascade stage on the current working sample set.
     * @param _featureEvaluator Feature evaluator already populated with images.
     * @param _numSamples Total number of samples (positives + negatives).
     * @param _precalcValBufSize Buffer size (MB) for cached feature values.
     * @param _precalcIdxBufSize Buffer size (MB) for cached sorted indices.
     * @param _params Boosting + cascade rate targets.
     * @return @c true once both rate targets are satisfied.
     */
    bool train( const CvFeatureEvaluator* _featureEvaluator,
                int _numSamples, int _precalcValBufSize, int _precalcIdxBufSize,
                const CvCascadeBoostParams& _params=CvCascadeBoostParams() );
    /// Evaluate the stage on sample @p sampleIdx. When @p returnSum is true
    /// returns the raw weak-response sum, otherwise the binary 0/1 decision.
    float predict( int sampleIdx, bool returnSum = false ) const;

    /// Decision threshold tuned to satisfy @c minHitRate.
    float getThreshold() const { return threshold; }
    /// Serialize the stage; @p featureMap remaps used feature indices to compact ids.
    void write( cv::FileStorage &fs, const cv::Mat& featureMap ) const;
    /// Restore the stage from a previously written XML node.
    bool read( const cv::FileNode &node, const CvFeatureEvaluator* _featureEvaluator,
               const CvCascadeBoostParams& _params );
    /// Mark every feature this stage references as used in @p featureMap.
    void markUsedFeaturesInMap( cv::Mat& featureMap );
private:
    /// Re-evaluate the stage on the working set and return @c true once
    /// the cascade rate targets are met.
    bool isErrDesired();
    bool set_params( const CvBoostParams& _params ) override;
    /// Update boosting sample weights after appending a new weak tree.
    void update_weights( CvBoostTree* tree );// override;

    float threshold;                  ///< Decision threshold tuned per stage.
    float minHitRate, maxFalseAlarm;  ///< Targets copied from CvCascadeBoostParams.
};

#endif
