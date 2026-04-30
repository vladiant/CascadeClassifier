/**
 * @file imagestorage.h
 * @brief Streaming readers for positive and negative training samples.
 *
 * @ref CvCascadeImageReader bundles two specialized streams used during
 * cascade training:
 *  - @c PosReader iterates through the binary @c .vec file produced by
 *    @c opencv_createsamples; each record is a fixed-size sub-window
 *    image of dimension @c winSize.
 *  - @c NegReader walks a list of background images, scanning each at
 *    multiple scales and offsets to mine sub-windows for use as negative
 *    samples.
 *
 * Both readers expose a uniform `bool get(cv::Mat&)` interface so the
 * trainer can request the next positive/negative without caring about
 * the underlying storage format.
 */

#ifndef _OPENCV_IMAGESTORAGE_H_
#define _OPENCV_IMAGESTORAGE_H_

#include <string>

#include <opencv2/core/core.hpp>

/**
 * @brief Pair of streaming sample readers used by the cascade trainer.
 *
 * After @ref create both @c posReader and @c negReader can be queried
 * via @ref getPos / @ref getNeg until they exhaust their respective
 * sources. @ref restart resets the positive stream so a new stage can
 * iterate it again from the beginning.
 */
class CvCascadeImageReader
{
public:
    /// Open both streams. @p _winSize is the cascade detection window.
    bool create( const std::string& _posFilename, const std::string& _negFilename, cv::Size _winSize );
    /// Rewind the positive stream to its first record.
    void restart() { posReader.restart(); }
    /// Fetch the next negative sub-window; returns @c false when exhausted.
    bool getNeg(cv::Mat &_img) { return negReader.get( _img ); }
    /// Fetch the next positive sub-window; returns @c false when exhausted.
    bool getPos(cv::Mat &_img) { return posReader.get( _img ); }

private:
    /// Reader over the binary @c .vec file produced by @c opencv_createsamples.
    class PosReader
    {
    public:
        PosReader();
        virtual ~PosReader();
        bool create( const std::string& _filename );
        bool get( cv::Mat &_img );
        void restart();

        short* vec;   ///< Buffer holding one decoded sample.
        FILE*  file;  ///< Underlying @c .vec file handle.
        int    count; ///< Number of records in the file.
        int    vecSize; ///< Pixels per record (= winSize.area()).
        int    last;  ///< Index of the last delivered record.
        int    base;  ///< Offset of the first record after the header.
    } posReader;

    /// Reader over a list of background images, sliding a sub-window
    /// over each at multiple scales.
    class NegReader
    {
    public:
        NegReader();
        bool create( const std::string& _filename, cv::Size _winSize );
        bool get( cv::Mat& _img );
        bool nextImg();

        cv::Mat     src, img;
        std::vector<std::string> imgFilenames; ///< Lines from the bg list file.
        cv::Point   offset, point;
        float   scale;
        float   scaleFactor; ///< Per-step scale multiplier (>1 for shrinking).
        float   stepFactor;  ///< Sliding-window step relative to @c winSize.
        size_t  last, round;
        cv::Size    winSize; ///< Detection window size copied from the cascade.
    } negReader;
};

#endif
