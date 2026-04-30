/**
 * @file lbpfeatures.h
 * @brief Local Binary Pattern (LBP) features for cascade training.
 *
 * Defines @ref CvLBPFeatureParams (a thin wrapper that fixes
 * @c maxCatCount = 256 — LBP is categorical) and @ref CvLBPEvaluator,
 * which computes 8-bit LBP codes by comparing the integral-image sum of
 * the central cell with each of its eight neighbours in a 3×3 grid of
 * equal-sized rectangles. Compared to Haar features LBP is faster to
 * evaluate but produces categorical responses (0..255).
 */

#ifndef _OPENCV_LBPFEATURES_H_
#define _OPENCV_LBPFEATURES_H_

#include "traincascade_features.h"

#define LBPF_NAME "lbpFeatureParams"

/// LBP-specific parameter struct; LBP responses are categorical so the
/// constructor sets @c maxCatCount appropriately.
struct CvLBPFeatureParams : CvFeatureParams
{
    CvLBPFeatureParams();

};

/**
 * @brief Evaluator for block-based LBP features.
 *
 * Each feature compares the central cell's pixel sum with its eight
 * neighbours in a 3x3 block grid; the outcome is encoded as an 8-bit
 * pattern. The class caches per-sample integral images in @c sum and
 * decodes feature geometry through @c features[featureIdx].
 */
class CvLBPEvaluator : public CvFeatureEvaluator
{
public:
    virtual ~CvLBPEvaluator() {}
    void init(const CvFeatureParams *_featureParams,
        int _maxSampleCount, cv::Size _winSize ) override;
    void setImage(const cv::Mat& img, uchar clsLabel, int idx) override;
    /// Return the 8-bit LBP code (0..255) cast to float.
    float operator()(int featureIdx, int sampleIdx) const override
    { return (float)features[featureIdx].calc( sum, sampleIdx); }
    void writeFeatures( cv::FileStorage &fs, const cv::Mat& featureMap ) const override;
protected:
    /// Enumerate every valid 3×3-block LBP feature for the configured window.
    void generateFeatures() override;

    /// Geometry of one LBP feature: rectangle covering the 3×3 grid plus
    /// 16 cached corner offsets (one per cell corner) into the integral image.
    class Feature
    {
    public:
        Feature();
        Feature( int offset, int x, int y, int _block_w, int _block_h  );
        /// Compute the 8-bit LBP code on sample row @p y.
        uchar calc( const cv::Mat& _sum, size_t y ) const;
        void write( cv::FileStorage &fs ) const;

        cv::Rect rect;
        int p[16]{};
    };
    std::vector<Feature> features; ///< Generated feature catalog.

    cv::Mat sum; ///< Cached integral images (one per sample, one per row).
};

inline uchar CvLBPEvaluator::Feature::calc(const cv::Mat &_sum, size_t y) const
{
    const int* psum = _sum.ptr<int>((int)y);
    int cval = psum[p[5]] - psum[p[6]] - psum[p[9]] + psum[p[10]];

    return (uchar)((psum[p[0]] - psum[p[1]] - psum[p[4]] + psum[p[5]] >= cval ? 128 : 0) |   // 0
        (psum[p[1]] - psum[p[2]] - psum[p[5]] + psum[p[6]] >= cval ? 64 : 0) |    // 1
        (psum[p[2]] - psum[p[3]] - psum[p[6]] + psum[p[7]] >= cval ? 32 : 0) |    // 2
        (psum[p[6]] - psum[p[7]] - psum[p[10]] + psum[p[11]] >= cval ? 16 : 0) |  // 5
        (psum[p[10]] - psum[p[11]] - psum[p[14]] + psum[p[15]] >= cval ? 8 : 0) | // 8
        (psum[p[9]] - psum[p[10]] - psum[p[13]] + psum[p[14]] >= cval ? 4 : 0) |  // 7
        (psum[p[8]] - psum[p[9]] - psum[p[12]] + psum[p[13]] >= cval ? 2 : 0) |   // 6
        (psum[p[4]] - psum[p[5]] - psum[p[8]] + psum[p[9]] >= cval ? 1 : 0));     // 3
}

#endif
