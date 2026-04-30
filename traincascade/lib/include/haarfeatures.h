/**
 * @file haarfeatures.h
 * @brief Haar-like rectangle features used by Viola–Jones-style cascades.
 *
 * Implements @ref CvHaarFeatureParams (with @c BASIC / @c CORE / @c ALL
 * feature-set modes) and @ref CvHaarEvaluator. A Haar feature is a sum of
 * up to @ref CV_HAAR_FEATURE_MAX weighted rectangles whose responses are
 * read from an integral image (and a 45°-tilted integral image when
 * @c tilted is set), then divided by a per-window normalization factor
 * to gain illumination invariance.
 */

#ifndef _OPENCV_HAARFEATURES_H_
#define _OPENCV_HAARFEATURES_H_

#include "traincascade_features.h"

/// Maximum number of weighted rectangles composing a single Haar feature.
#define CV_HAAR_FEATURE_MAX      3

#define HFP_NAME "haarFeatureParams"

/**
 * @brief Parameters specific to Haar feature generation.
 *
 * @c mode picks the feature catalog density:
 *  - @c BASIC: only upright Viola–Jones features;
 *  - @c CORE:  full upright catalog;
 *  - @c ALL:   upright + 45°-tilted features (largest catalog).
 */
class CvHaarFeatureParams : public CvFeatureParams
{
public:
    enum { BASIC = 0, CORE = 1, ALL = 2 };
     /* 0 - BASIC = Viola
     *  1 - CORE  = All upright
     *  2 - ALL   = All features */

    CvHaarFeatureParams();
    CvHaarFeatureParams( int _mode );

    void init( const CvFeatureParams& fp ) override;
    void write( cv::FileStorage &fs ) const override;
    bool read( const cv::FileNode &node ) override;

    void printDefaults() const override;
    void printAttrs() const override;
    bool scanAttr( const std::string prm, const std::string val) override;

    int mode; ///< Feature-catalog density: BASIC, CORE or ALL.
};

/**
 * @brief Evaluator that computes Haar feature responses from integral images.
 *
 * On @ref setImage the evaluator stores the sample's integral image in
 * @c sum (and tilted integral image in @c tilted when needed) plus the
 * normalization factor in @c normfactor. @ref operator() then evaluates
 * the geometry stored in @c features[featureIdx] in O(rects) time.
 */
class CvHaarEvaluator : public CvFeatureEvaluator
{
public:
    void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize ) override;
    void setImage(const cv::Mat& img, uchar clsLabel, int idx) override;
    float operator()(int featureIdx, int sampleIdx) const override;
    void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const override;
    /// Write a single feature using the legacy XML schema (for compatibility).
    void writeFeature( cv::FileStorage &fs, int fi ) const; // for old file fornat
protected:
    /// Enumerate every valid Haar feature for the configured @c winSize / mode.
    void generateFeatures() override;

    /// Geometry of one Haar feature: up to three weighted rectangles, optionally tilted.
    class Feature
    {
    public:
        Feature();
        Feature( int offset, bool _tilted,
            int x0, int y0, int w0, int h0, float wt0,
            int x1, int y1, int w1, int h1, float wt1,
            int x2 = 0, int y2 = 0, int w2 = 0, int h2 = 0, float wt2 = 0.0F );
        /// Compute the unnormalized feature response for sample @p y.
        float calc( const cv::Mat &sum, const cv::Mat &tilted, size_t y) const;
        void write( cv::FileStorage &fs ) const;

        bool  tilted; ///< @c true when reading from the tilted integral image.
        struct
        {
            cv::Rect r;
            float weight;
        } rect[CV_HAAR_FEATURE_MAX];

        /// Precomputed corner offsets for each rectangle into the row-stride flattening.
        struct
        {
            int p0, p1, p2, p3;
        } fastRect[CV_HAAR_FEATURE_MAX]{};
    };

    std::vector<Feature> features; ///< Generated feature catalog.
    cv::Mat  sum;         /* sum images (each row represents image) */
    cv::Mat  tilted;      /* tilted sum images (each row represents image) */
    cv::Mat  normfactor;  /* normalization factor */
};

inline float CvHaarEvaluator::operator()(int featureIdx, int sampleIdx) const
{
    float nf = normfactor.at<float>(0, sampleIdx);
    return !nf ? 0.0f : (features[featureIdx].calc( sum, tilted, sampleIdx)/nf);
}

inline float CvHaarEvaluator::Feature::calc( const cv::Mat &_sum, const cv::Mat &_tilted, size_t y) const
{
    const int* img = tilted ? _tilted.ptr<int>((int)y) : _sum.ptr<int>((int)y);
    float ret = rect[0].weight * (img[fastRect[0].p0] - img[fastRect[0].p1] - img[fastRect[0].p2] + img[fastRect[0].p3] ) +
        rect[1].weight * (img[fastRect[1].p0] - img[fastRect[1].p1] - img[fastRect[1].p2] + img[fastRect[1].p3] );
    if( rect[2].weight != 0.0f )
        ret += rect[2].weight * (img[fastRect[2].p0] - img[fastRect[2].p1] - img[fastRect[2].p2] + img[fastRect[2].p3] );
    return ret;
}

#endif
