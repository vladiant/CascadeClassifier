/**
 * @file traincascade_features.h
 * @brief Common feature-evaluation infrastructure shared by Haar, LBP and HOG.
 *
 * Declares:
 *  - @ref CvParams, the polymorphic base for every parameter struct that
 *    can be loaded from / saved to a @c FileStorage and parsed from
 *    command-line attributes.
 *  - @ref CvFeatureParams, the family-agnostic feature parameter struct
 *    plus a factory that instantiates HAAR/LBP/HOG variants on demand.
 *  - @ref CvFeatureEvaluator, the abstract evaluator used by the boosting
 *    trainer to read the response of feature @p featureIdx on sample
 *    @p sampleIdx without exposing the underlying integral images.
 *
 * The macros @c CV_SUM_OFFSETS / @c CV_TILTED_OFFSETS and the helper
 * @ref calcNormFactor are used by the concrete evaluators to convert
 * (x, y, w, h) rectangles into linear offsets into a precomputed integral
 * image, which is the bottleneck operation of every weak-classifier
 * evaluation.
 */

#ifndef _OPENCV_FEATURES_H_
#define _OPENCV_FEATURES_H_

#include "imagestorage.h"

#include <stdio.h>

/// XML tag under which the list of features is serialized in @c cascade.xml.
#define FEATURES "features"

/**
 * @brief Compute the four corner offsets of an axis-aligned rectangle in an
 *        integral image laid out with row stride @p step.
 *
 * Given an integral image @c S the sum of pixels inside @p rect can be
 * recovered as @c S[p3] - S[p1] - S[p2] + S[p0], so feature evaluators
 * cache the four offsets up front and reuse them per sample.
 */
template <typename T, typename Rect>
void CV_SUM_OFFSETS(T& p0, T& p1, T& p2, T& p3, const Rect& rect, T step ) {                     
    /* (x, y) */                                                          
    p0 = rect.x + step * rect.y;                                  
    /* (x + w, y) */                                                      
    p1 = rect.x + (rect).width + step * rect.y;                   
    /* (x + w, y) */                                                      
    p2 = rect.x + step * (rect.y + rect.height);                
    /* (x + w, y + h) */                                                  
    p3 = rect.x + rect.width + step * (rect.y + rect.height);
}

/// Same as @ref CV_SUM_OFFSETS but for a 45°-rotated (tilted) rectangle.
/// Tilted Haar features rely on a separately-computed tilted integral image.
#define CV_TILTED_OFFSETS( p0, p1, p2, p3, rect, step )                   \
    /* (x, y) */                                                          \
    (p0) = (rect).x + (step) * (rect).y;                                  \
    /* (x - h, y + h) */                                                  \
    (p1) = (rect).x - (rect).height + (step) * ((rect).y + (rect).height);\
    /* (x + w, y + w) */                                                  \
    (p2) = (rect).x + (rect).width + (step) * ((rect).y + (rect).width);  \
    /* (x + w - h, y + w + h) */                                          \
    (p3) = (rect).x + (rect).width - (rect).height                        \
           + (step) * ((rect).y + (rect).width + (rect).height);

/**
 * @brief Compute the per-window normalization factor used by Haar features.
 *
 * Returns @f$\sqrt{N \cdot \text{sqSum} - \text{sum}^2}@f$ for the
 * detection window covered by the integral images @p sum and @p sqSum.
 * Haar feature responses are divided by this factor to be invariant to
 * window-level brightness and contrast.
 */
float calcNormFactor( const cv::Mat& sum, const cv::Mat& sqSum );

/**
 * @brief Serialize the subset of @p features marked as used in @p featureMap.
 *
 * @p featureMap is a 1xN row mapping each global feature index to its
 * compact index (or -1 if the cascade never selected it). Skipping unused
 * features keeps the output @c cascade.xml small.
 */
template<class Feature>
void _writeFeatures( const std::vector<Feature> features, cv::FileStorage &fs, const cv::Mat& featureMap )
{
    fs << FEATURES << "[";
    const cv::Mat_<int>& featureMap_ = (const cv::Mat_<int>&)featureMap;
    for ( int fi = 0; fi < featureMap.cols; fi++ )
        if ( featureMap_(0, fi) >= 0 )
        {
            fs << "{";
            features[fi].write( fs );
            fs << "}";
        }
    fs << "]";
}

/**
 * @brief Polymorphic base for parameter structs that round-trip to disk.
 *
 * Every concrete parameter struct in the trainer (cascade-level, feature,
 * boosting, etc.) inherits @ref CvParams so the driver can iterate them
 * generically when reading/writing @c params.xml or scanning command-line
 * attributes.
 */
class CvParams
{
public:
    CvParams();
    virtual ~CvParams() {}
    /// Persist to @p fs (called inside an open mapping node).
    virtual void write( cv::FileStorage &fs ) const = 0;
    /// Restore from @p node; @c false on malformed input.
    virtual bool read( const cv::FileNode &node ) = 0;
    /// Print the default value of every parameter to stdout.
    virtual void printDefaults() const;
    /// Print the currently configured value of every parameter to stdout.
    virtual void printAttrs() const;
    /// Apply a single command-line attribute (e.g. @c -w 24); returns
    /// @c false when the attribute name is not recognized.
    virtual bool scanAttr( const std::string prmName, const std::string val );
    std::string name; ///< Human-readable name used as the FileStorage tag.
};

/**
 * @brief Family-agnostic parameter struct for feature evaluators.
 *
 * Holds the descriptor sizes shared across feature families and provides
 * a static @ref create factory that instantiates the matching subclass
 * (@c CvHaarFeatureParams, @c CvLBPFeatureParams or @c CvHOGFeatureParams).
 */
class CvFeatureParams : public CvParams
{
public:
    enum { HAAR = 0, LBP = 1, HOG = 2 }; ///< Supported feature families.
    CvFeatureParams();
    /// Copy field-by-field from another instance (used to clone defaults).
    virtual void init( const CvFeatureParams& fp );
    void write( cv::FileStorage &fs ) const override;
    bool read( const cv::FileNode &node ) override;
    /// Factory: instantiate the parameters subclass matching @p featureType.
    static cv::Ptr<CvFeatureParams> create( int featureType );
    int maxCatCount; ///< Number of categorical bins (0 for purely numerical features such as Haar).
    int featSize;    ///< Feature descriptor length: 1 for HAAR/LBP, @c N_BINS*N_CELLS for HOG.
};

/**
 * @brief Abstract evaluator that returns feature responses on demand.
 *
 * The boosting trainer never sees the underlying integral images directly
 * — it only calls @ref operator() with an @em index pair and gets back a
 * float. Concrete subclasses (@c CvHaarEvaluator, @c CvLBPEvaluator,
 * @c CvHOGEvaluator) cache per-sample integral images inside @c setImage
 * and decode @p featureIdx via their own feature catalog.
 */
class CvFeatureEvaluator
{
public:
    virtual ~CvFeatureEvaluator() {}
    /// Allocate per-sample buffers for at most @p _maxSampleCount samples
    /// of size @p _winSize and generate the feature catalog.
    virtual void init(const CvFeatureParams *_featureParams,
                      int _maxSampleCount, cv::Size _winSize );
    /// Cache integral images for sample @p idx; @p clsLabel is the binary
    /// class label (1 for positives, 0 for negatives).
    virtual void setImage(const cv::Mat& img, uchar clsLabel, int idx);
    /// Serialize every selected feature's geometry (used at save time).
    virtual void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const = 0;
    /// Return the response of feature @p featureIdx on sample @p sampleIdx.
    virtual float operator()(int featureIdx, int sampleIdx) const = 0;
    /// Factory: instantiate the evaluator matching @p type (HAAR/LBP/HOG).
    static cv::Ptr<CvFeatureEvaluator> create(int type);

    int getNumFeatures() const { return numFeatures; }
    int getMaxCatCount() const { return featureParams->maxCatCount; }
    int getFeatureSize() const { return featureParams->featSize; }
    /// Cached class-label vector (1 = positive, 0 = negative) per sample.
    const cv::Mat& getCls() const { return cls; }
    float getCls(int si) const { return cls.at<float>(si, 0); }
protected:
    /// Build the feature catalog for the current window size.
    virtual void generateFeatures() = 0;

    int npos, nneg;                  ///< Number of positives / negatives currently cached.
    int numFeatures;                 ///< Size of the feature catalog.
    cv::Size winSize;                ///< Sub-window size for which features were generated.
    CvFeatureParams *featureParams;  ///< Non-owning pointer to feature parameters.
    cv::Mat cls;                     ///< Per-sample class labels (Nx1 CV_32F).
};

#endif
