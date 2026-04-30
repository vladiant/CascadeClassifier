/**
 * @file traincascade.cpp
 * @brief Command-line entry point for the cascade classifier trainer.
 *
 * This is the @c traincascade executable. It instantiates a
 * @ref CvCascadeClassifier, parses command-line attributes into the
 * three nested parameter structs (@ref CvCascadeParams,
 * @ref CvCascadeBoostParams, @ref CvFeatureParams) and dispatches to
 * @ref CvCascadeClassifier::train. Unknown attributes are forwarded to
 * each parameter struct in turn via @c scanAttr; the first one that
 * recognizes the flag consumes the value.
 *
 * Required arguments:
 *  - @c -data <dir>: output directory for stage XMLs and @c cascade.xml.
 *  - @c -vec  <file>: positives, produced by @c opencv_createsamples.
 *  - @c -bg   <file>: text file listing background images for negatives.
 *
 * Run the binary without arguments for the full list of optional flags
 * and their default values.
 */
#include <iostream>

#include <opencv2/core.hpp>

#include "cascadeclassifier.h"

using namespace std;
using namespace cv;

/**
 * @brief Program entry point: parse CLI flags and run cascade training.
 *
 * The argument-parsing loop tries known top-level flags first, then
 * delegates unknown @c -name value pairs to the parameter structs
 * (@c cascadeParams, @c stageParams, then each entry of @c featureParams)
 * so feature-family-specific flags such as @c -mode (Haar) or @c -mode
 * (LBP) are picked up by the matching evaluator's parameters.
 */
int main( int argc, char* argv[] )
{
    /// Cascade trainer; populates output directory with stage XMLs.
    CvCascadeClassifier classifier;
    string cascadeDirName, vecName, bgName;
    int numPos    = 2000;     ///< Default positives consumed per stage.
    int numNeg    = 1000;     ///< Default negatives mined per stage.
    int numStages = 20;       ///< Maximum number of cascade stages.
    int numThreads = getNumThreads();
    int precalcValBufSize = 1024,  ///< Feature-value cache size in MB.
        precalcIdxBufSize = 1024;  ///< Sorted-index cache size in MB.
    bool baseFormatSave = false;   ///< When true also write the legacy XML layout.
    double acceptanceRatioBreakValue = -1.0; ///< -1 disables early stopping.

    // The three parameter structs collectively configure every aspect of
    // the cascade: window size & feature type (cascadeParams), boosting
    // and stage rate targets (stageParams), and feature-specific knobs
    // (featureParams[HAAR / LBP / HOG]).
    CvCascadeParams cascadeParams;
    CvCascadeBoostParams stageParams;
    Ptr<CvFeatureParams> featureParams[] = { makePtr<CvHaarFeatureParams>(),
                                             makePtr<CvLBPFeatureParams>(),
                                             makePtr<CvHOGFeatureParams>()
                                           };
    int fc = sizeof(featureParams)/sizeof(featureParams[0]);
    if( argc == 1 )
    {
        cout << "Usage: " << argv[0] << endl;
        cout << "  -data <cascade_dir_name>" << endl;
        cout << "  -vec <vec_file_name>" << endl;
        cout << "  -bg <background_file_name>" << endl;
        cout << "  [-numPos <number_of_positive_samples = " << numPos << ">]" << endl;
        cout << "  [-numNeg <number_of_negative_samples = " << numNeg << ">]" << endl;
        cout << "  [-numStages <number_of_stages = " << numStages << ">]" << endl;
        cout << "  [-precalcValBufSize <precalculated_vals_buffer_size_in_Mb = " << precalcValBufSize << ">]" << endl;
        cout << "  [-precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb = " << precalcIdxBufSize << ">]" << endl;
        cout << "  [-baseFormatSave]" << endl;
        cout << "  [-numThreads <max_number_of_threads = " << numThreads << ">]" << endl;
        cout << "  [-acceptanceRatioBreakValue <value> = " << acceptanceRatioBreakValue << ">]" << endl;
        cascadeParams.printDefaults();
        stageParams.printDefaults();
        for( int fi = 0; fi < fc; fi++ )
            featureParams[fi]->printDefaults();
        return 0;
    }

    for( int i = 1; i < argc; i++ )
    {
        // Try every known top-level flag in order; if none match, fall
        // through to the parameter structs which expose their own
        // attributes via scanAttr().
        bool set = false;
        if( !strcmp( argv[i], "-data" ) )
        {
            cascadeDirName = argv[++i];
        }
        else if( !strcmp( argv[i], "-vec" ) )
        {
            vecName = argv[++i];
        }
        else if( !strcmp( argv[i], "-bg" ) )
        {
            bgName = argv[++i];
        }
        else if( !strcmp( argv[i], "-numPos" ) )
        {
            numPos = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-numNeg" ) )
        {
            numNeg = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-numStages" ) )
        {
            numStages = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-precalcValBufSize" ) )
        {
            precalcValBufSize = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-precalcIdxBufSize" ) )
        {
            precalcIdxBufSize = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-baseFormatSave" ) )
        {
            baseFormatSave = true;
        }
        else if( !strcmp( argv[i], "-numThreads" ) )
        {
          numThreads = atoi(argv[++i]);
        }
        else if( !strcmp( argv[i], "-acceptanceRatioBreakValue" ) )
        {
          acceptanceRatioBreakValue = atof(argv[++i]);
        }
        else if ( cascadeParams.scanAttr( argv[i], argv[i+1] ) ) { i++; }
        else if ( stageParams.scanAttr( argv[i], argv[i+1] ) ) { i++; }
        else if ( !set )
        {
            for( int fi = 0; fi < fc; fi++ )
            {
                set = featureParams[fi]->scanAttr(argv[i], argv[i+1]);
                if ( !set )
                {
                    i++;
                    break;
                }
            }
        }
    }

    setNumThreads( numThreads );
    // Hand control over to the cascade trainer; it owns the multi-stage
    // training loop, sample mining, and persistence of intermediate state.
    classifier.train( cascadeDirName,
                      vecName,
                      bgName,
                      numPos, numNeg,
                      precalcValBufSize, precalcIdxBufSize,
                      numStages,
                      cascadeParams,
                      *featureParams[cascadeParams.featureType],
                      stageParams,
                      baseFormatSave,
                      acceptanceRatioBreakValue );
    return 0;
}
