#include "o_cvcascadeboosttraindata.h"

#include <iostream>

#include <opencv2/core/core.hpp>

#include "cascadeclassifier.h"

static const int BlockSizeDelta = 1 << 10;
static const int MinBlockSize = 1 << 16;

// TODO: Duplicated
template<typename T, typename Idx>
class LessThanIdx
{
public:
    LessThanIdx( const T* _arr ) : arr(_arr) {}
    bool operator()(Idx a, Idx b) const { return arr[a] < arr[b]; }
    const T* arr;
};

// TODO: Duplicated
static inline int cvAlign( int size, int align )
{
    CV_DbgAssert( (align & (align-1)) == 0 && size < INT_MAX );
    return (size + align - 1) & -align;
}

CvDTreeNode* CvCascadeBoostTrainData::subsample_data( const CvMat* _subsample_idx )
{
    CvDTreeNode* root = 0;
    CvMat* isubsample_idx = 0;
    CvMat* subsample_co = 0;

    bool isMakeRootCopy = true;

    if( !data_root )
        CV_Error( CV_StsError, "No training data has been set" );

    if( _subsample_idx )
    {
        CV_Assert( (isubsample_idx = cvPreprocessIndexArray( _subsample_idx, sample_count )) != 0 );

        if( isubsample_idx->cols + isubsample_idx->rows - 1 == sample_count )
        {
            const int* sidx = isubsample_idx->data.i;
            for( int i = 0; i < sample_count; i++ )
            {
                if( sidx[i] != i )
                {
                    isMakeRootCopy = false;
                    break;
                }
            }
        }
        else
            isMakeRootCopy = false;
    }

    if( isMakeRootCopy )
    {
        // make a copy of the root node
        CvDTreeNode temp;
        int i;
        root = new_node( 0, 1, 0, 0 );
        temp = *root;
        *root = *data_root;
        root->num_valid = temp.num_valid;
        if( root->num_valid )
        {
            for( i = 0; i < var_count; i++ )
                root->num_valid[i] = data_root->num_valid[i];
        }
        root->cv_Tn = temp.cv_Tn;
        root->cv_node_risk = temp.cv_node_risk;
        root->cv_node_error = temp.cv_node_error;
    }
    else
    {
        int* sidx = isubsample_idx->data.i;
        // co - array of count/offset pairs (to handle duplicated values in _subsample_idx)
        int* co, cur_ofs = 0;
        int workVarCount = get_work_var_count();
        int count = isubsample_idx->rows + isubsample_idx->cols - 1;

        root = new_node( 0, count, 1, 0 );

        CV_Assert( (subsample_co = cvCreateMat( 1, sample_count*2, CV_32SC1 )) != 0);
        cvZero( subsample_co );
        co = subsample_co->data.i;
        for( int i = 0; i < count; i++ )
            co[sidx[i]*2]++;
        for( int i = 0; i < sample_count; i++ )
        {
            if( co[i*2] )
            {
                co[i*2+1] = cur_ofs;
                cur_ofs += co[i*2];
            }
            else
                co[i*2+1] = -1;
        }

        cv::AutoBuffer<uchar> inn_buf(sample_count*(2*sizeof(int) + sizeof(float)));
        // subsample ordered variables
        for( int vi = 0; vi < numPrecalcIdx; vi++ )
        {
            int ci = get_var_type(vi);
            CV_Assert( ci < 0 );

            int *src_idx_buf = (int*)inn_buf.data();
            float *src_val_buf = (float*)(src_idx_buf + sample_count);
            int* sample_indices_buf = (int*)(src_val_buf + sample_count);
            const int* src_idx = 0;
            const float* src_val = 0;
            get_ord_var_data( data_root, vi, src_val_buf, src_idx_buf, &src_val, &src_idx, sample_indices_buf );

            int j = 0, idx, count_i;
            int num_valid = data_root->get_num_valid(vi);
            CV_Assert( num_valid == sample_count );

            if (is_buf_16u)
            {
                unsigned short* udst_idx = (unsigned short*)(buf->data.s + root->buf_idx*get_length_subbuf() +
                    (size_t)vi*sample_count + data_root->offset);
                for( int i = 0; i < num_valid; i++ )
                {
                    idx = src_idx[i];
                    count_i = co[idx*2];
                    if( count_i )
                        for( cur_ofs = co[idx*2+1]; count_i > 0; count_i--, j++, cur_ofs++ )
                            udst_idx[j] = (unsigned short)cur_ofs;
                }
            }
            else
            {
                int* idst_idx = buf->data.i + root->buf_idx*get_length_subbuf() +
                    (size_t)vi*sample_count + root->offset;
                for( int i = 0; i < num_valid; i++ )
                {
                    idx = src_idx[i];
                    count_i = co[idx*2];
                    if( count_i )
                        for( cur_ofs = co[idx*2+1]; count_i > 0; count_i--, j++, cur_ofs++ )
                            idst_idx[j] = cur_ofs;
                }
            }
        }

        // subsample cv_lables
        const int* src_lbls = get_cv_labels(data_root, (int*)inn_buf.data());
        if (is_buf_16u)
        {
            unsigned short* udst = (unsigned short*)(buf->data.s + root->buf_idx*get_length_subbuf() +
                (size_t)(workVarCount-1)*sample_count + root->offset);
            for( int i = 0; i < count; i++ )
                udst[i] = (unsigned short)src_lbls[sidx[i]];
        }
        else
        {
            int* idst = buf->data.i + root->buf_idx*get_length_subbuf() +
                (size_t)(workVarCount-1)*sample_count + root->offset;
            for( int i = 0; i < count; i++ )
                idst[i] = src_lbls[sidx[i]];
        }

        // subsample sample_indices
        const int* sample_idx_src = get_sample_indices(data_root, (int*)inn_buf.data());
        if (is_buf_16u)
        {
            unsigned short* sample_idx_dst = (unsigned short*)(buf->data.s + root->buf_idx*get_length_subbuf() +
                (size_t)workVarCount*sample_count + root->offset);
            for( int i = 0; i < count; i++ )
                sample_idx_dst[i] = (unsigned short)sample_idx_src[sidx[i]];
        }
        else
        {
            int* sample_idx_dst = buf->data.i + root->buf_idx*get_length_subbuf() +
                (size_t)workVarCount*sample_count + root->offset;
            for( int i = 0; i < count; i++ )
                sample_idx_dst[i] = sample_idx_src[sidx[i]];
        }

        for( int vi = 0; vi < var_count; vi++ )
            root->set_num_valid(vi, count);
    }

    cvReleaseMat( &isubsample_idx );
    cvReleaseMat( &subsample_co );

    return root;
}

//---------------------------- CascadeBoostTrainData -----------------------------

CvCascadeBoostTrainData::CvCascadeBoostTrainData( const CvFeatureEvaluator* _featureEvaluator,
                                                  const CvDTreeParams& _params )
{
    is_classifier = true;
    var_all = var_count = (int)_featureEvaluator->getNumFeatures();

    featureEvaluator = _featureEvaluator;
    shared = true;
    set_params( _params );
    max_c_count = MAX( 2, featureEvaluator->getMaxCatCount() );
    var_type = cvCreateMat( 1, var_count + 2, CV_32SC1 );
    if ( featureEvaluator->getMaxCatCount() > 0 )
    {
        numPrecalcIdx = 0;
        cat_var_count = var_count;
        ord_var_count = 0;
        for( int vi = 0; vi < var_count; vi++ )
        {
            var_type->data.i[vi] = vi;
        }
    }
    else
    {
        cat_var_count = 0;
        ord_var_count = var_count;
        for( int vi = 1; vi <= var_count; vi++ )
        {
            var_type->data.i[vi-1] = -vi;
        }
    }
    var_type->data.i[var_count] = cat_var_count;
    var_type->data.i[var_count+1] = cat_var_count+1;

    int maxSplitSize = cvAlign(sizeof(CvDTreeSplit) + (MAX(0,max_c_count - 33)/32)*sizeof(int),sizeof(void*));
    int treeBlockSize = MAX((int)sizeof(CvDTreeNode)*8, maxSplitSize);
    treeBlockSize = MAX(treeBlockSize + BlockSizeDelta, MinBlockSize);
    tree_storage = cvCreateMemStorage( treeBlockSize );
    node_heap = cvCreateSet( 0, sizeof(node_heap[0]), sizeof(CvDTreeNode), tree_storage );
    split_heap = cvCreateSet( 0, sizeof(split_heap[0]), maxSplitSize, tree_storage );
}

CvCascadeBoostTrainData::CvCascadeBoostTrainData( const CvFeatureEvaluator* _featureEvaluator,
                                                 int _numSamples,
                                                 int _precalcValBufSize, int _precalcIdxBufSize,
                                                 const CvDTreeParams& _params )
{
    setData( _featureEvaluator, _numSamples, _precalcValBufSize, _precalcIdxBufSize, _params );
}

void CvCascadeBoostTrainData::setData( const CvFeatureEvaluator* _featureEvaluator,
                                      int _numSamples,
                                      int _precalcValBufSize, int _precalcIdxBufSize,
                                      const CvDTreeParams& _params )
{
    int* idst = 0;
    unsigned short* udst = 0;

    uint64 effective_buf_size = 0;
    int effective_buf_height = 0, effective_buf_width = 0;


    clear();
    shared = true;
    have_labels = true;
    have_priors = false;
    is_classifier = true;

    rng = &cv::theRNG();

    set_params( _params );

    CV_Assert( _featureEvaluator );
    featureEvaluator = _featureEvaluator;

    max_c_count = MAX( 2, featureEvaluator->getMaxCatCount() );
    _resp = cvMat(featureEvaluator->getCls());
    responses = &_resp;
    // TODO: check responses: elements must be 0 or 1

    if( _precalcValBufSize < 0 || _precalcIdxBufSize < 0)
        CV_Error( CV_StsOutOfRange, "_numPrecalcVal and _numPrecalcIdx must be positive or 0" );

    var_count = var_all = featureEvaluator->getNumFeatures() * featureEvaluator->getFeatureSize();
    sample_count = _numSamples;

    is_buf_16u = false;
    if (sample_count < 65536)
        is_buf_16u = true;

    numPrecalcVal = std::min( cvRound((double)_precalcValBufSize*1048576. / (sizeof(float)*sample_count)), var_count );
    numPrecalcIdx = std::min( cvRound((double)_precalcIdxBufSize*1048576. /
                ((is_buf_16u ? sizeof(unsigned short) : sizeof (int))*sample_count)), var_count );

    assert( numPrecalcIdx >= 0 && numPrecalcVal >= 0 );

    valCache.create( numPrecalcVal, sample_count, CV_32FC1 );
    var_type = cvCreateMat( 1, var_count + 2, CV_32SC1 );

    if ( featureEvaluator->getMaxCatCount() > 0 )
    {
        numPrecalcIdx = 0;
        cat_var_count = var_count;
        ord_var_count = 0;
        for( int vi = 0; vi < var_count; vi++ )
        {
            var_type->data.i[vi] = vi;
        }
    }
    else
    {
        cat_var_count = 0;
        ord_var_count = var_count;
        for( int vi = 1; vi <= var_count; vi++ )
        {
            var_type->data.i[vi-1] = -vi;
        }
    }
    var_type->data.i[var_count] = cat_var_count;
    var_type->data.i[var_count+1] = cat_var_count+1;
    work_var_count = ( cat_var_count ? 0 : numPrecalcIdx ) + 1/*cv_lables*/;
    buf_count = 2;

    buf_size = -1; // the member buf_size is obsolete

    effective_buf_size = (uint64)(work_var_count + 1)*(uint64)sample_count * buf_count; // this is the total size of "CvMat buf" to be allocated
    effective_buf_width = sample_count;
    effective_buf_height = work_var_count+1;

    if (effective_buf_width >= effective_buf_height)
        effective_buf_height *= buf_count;
    else
        effective_buf_width *= buf_count;

    if ((uint64)effective_buf_width * (uint64)effective_buf_height != effective_buf_size)
    {
        CV_Error(CV_StsBadArg, "The memory buffer cannot be allocated since its size exceeds integer fields limit");
    }

    if ( is_buf_16u )
        buf = cvCreateMat( effective_buf_height, effective_buf_width, CV_16UC1 );
    else
        buf = cvCreateMat( effective_buf_height, effective_buf_width, CV_32SC1 );

    cat_count = cvCreateMat( 1, cat_var_count + 1, CV_32SC1 );

    // precalculate valCache and set indices in buf
    precalculate();

    // now calculate the maximum size of split,
    // create memory storage that will keep nodes and splits of the decision tree
    // allocate root node and the buffer for the whole training data
    int maxSplitSize = cvAlign(sizeof(CvDTreeSplit) +
        (MAX(0,sample_count - 33)/32)*sizeof(int),sizeof(void*));
    int treeBlockSize = MAX((int)sizeof(CvDTreeNode)*8, maxSplitSize);
    treeBlockSize = MAX(treeBlockSize + BlockSizeDelta, MinBlockSize);
    tree_storage = cvCreateMemStorage( treeBlockSize );
    node_heap = cvCreateSet( 0, sizeof(*node_heap), sizeof(CvDTreeNode), tree_storage );

    int nvSize = var_count*sizeof(int);
    nvSize = cvAlign(MAX( nvSize, (int)sizeof(CvSetElem) ), sizeof(void*));
    int tempBlockSize = nvSize;
    tempBlockSize = MAX( tempBlockSize + BlockSizeDelta, MinBlockSize );
    temp_storage = cvCreateMemStorage( tempBlockSize );
    nv_heap = cvCreateSet( 0, sizeof(*nv_heap), nvSize, temp_storage );

    data_root = new_node( 0, sample_count, 0, 0 );

    // set sample labels
    if (is_buf_16u)
        udst = (unsigned short*)(buf->data.s + (size_t)work_var_count*sample_count);
    else
        idst = buf->data.i + (size_t)work_var_count*sample_count;

    for (int si = 0; si < sample_count; si++)
    {
        if (udst)
            udst[si] = (unsigned short)si;
        else
            idst[si] = si;
    }
    for( int vi = 0; vi < var_count; vi++ )
        data_root->set_num_valid(vi, sample_count);
    for( int vi = 0; vi < cat_var_count; vi++ )
        cat_count->data.i[vi] = max_c_count;

    cat_count->data.i[cat_var_count] = 2;

    maxSplitSize = cvAlign(sizeof(CvDTreeSplit) +
        (MAX(0,max_c_count - 33)/32)*sizeof(int),sizeof(void*));
    split_heap = cvCreateSet( 0, sizeof(*split_heap), maxSplitSize, tree_storage );

    priors = cvCreateMat( 1, get_num_classes(), CV_64F );
    cvSet(priors, cvScalar(1));
    priors_mult = cvCloneMat( priors );
    counts = cvCreateMat( 1, get_num_classes(), CV_32SC1 );
    direction = cvCreateMat( 1, sample_count, CV_8UC1 );
    split_buf = cvCreateMat( 1, sample_count, CV_32SC1 );//TODO: make a pointer
}

void CvCascadeBoostTrainData::free_train_data()
{
    CvDTreeTrainData::free_train_data();
    valCache.release();
}

const int* CvCascadeBoostTrainData::get_class_labels( CvDTreeNode* n, int* labelsBuf)
{
    int nodeSampleCount = n->sample_count;
    int rStep = CV_IS_MAT_CONT( responses->type ) ? 1 : responses->step / CV_ELEM_SIZE( responses->type );

    int* sampleIndicesBuf = labelsBuf; //
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);
    for( int si = 0; si < nodeSampleCount; si++ )
    {
        int sidx = sampleIndices[si];
        labelsBuf[si] = (int)responses->data.fl[sidx*rStep];
    }
    return labelsBuf;
}

const int* CvCascadeBoostTrainData::get_sample_indices( CvDTreeNode* n, int* indicesBuf )
{
    return CvDTreeTrainData::get_cat_var_data( n, get_work_var_count(), indicesBuf );
}

const int* CvCascadeBoostTrainData::get_cv_labels( CvDTreeNode* n, int* labels_buf )
{
    return CvDTreeTrainData::get_cat_var_data( n, get_work_var_count() - 1, labels_buf );
}

void CvCascadeBoostTrainData::get_ord_var_data( CvDTreeNode* n, int vi, float* ordValuesBuf, int* sortedIndicesBuf,
        const float** ordValues, const int** sortedIndices, int* sampleIndicesBuf )
{
    int nodeSampleCount = n->sample_count;
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);

    if ( vi < numPrecalcIdx )
    {
        if( !is_buf_16u )
            *sortedIndices = buf->data.i + n->buf_idx*get_length_subbuf() + (size_t)vi*sample_count + n->offset;
        else
        {
            const unsigned short* shortIndices = (const unsigned short*)(buf->data.s + n->buf_idx*get_length_subbuf() +
                                                    (size_t)vi*sample_count + n->offset );
            for( int i = 0; i < nodeSampleCount; i++ )
                sortedIndicesBuf[i] = shortIndices[i];

            *sortedIndices = sortedIndicesBuf;
        }

        if( vi < numPrecalcVal )
        {
            for( int i = 0; i < nodeSampleCount; i++ )
            {
                int idx = (*sortedIndices)[i];
                idx = sampleIndices[idx];
                ordValuesBuf[i] =  valCache.at<float>( vi, idx);
            }
        }
        else
        {
            for( int i = 0; i < nodeSampleCount; i++ )
            {
                int idx = (*sortedIndices)[i];
                idx = sampleIndices[idx];
                ordValuesBuf[i] = (*featureEvaluator)( vi, idx);
            }
        }
    }
    else // vi >= numPrecalcIdx
    {
        cv::AutoBuffer<float> abuf(nodeSampleCount);
        float* sampleValues = &abuf[0];

        if ( vi < numPrecalcVal )
        {
            for( int i = 0; i < nodeSampleCount; i++ )
            {
                sortedIndicesBuf[i] = i;
                sampleValues[i] = valCache.at<float>( vi, sampleIndices[i] );
            }
        }
        else
        {
            for( int i = 0; i < nodeSampleCount; i++ )
            {
                sortedIndicesBuf[i] = i;
                sampleValues[i] = (*featureEvaluator)( vi, sampleIndices[i]);
            }
        }
        std::sort(sortedIndicesBuf, sortedIndicesBuf + nodeSampleCount, LessThanIdx<float, int>(&sampleValues[0]) );
        for( int i = 0; i < nodeSampleCount; i++ )
            ordValuesBuf[i] = (&sampleValues[0])[sortedIndicesBuf[i]];
        *sortedIndices = sortedIndicesBuf;
    }

    *ordValues = ordValuesBuf;
}

const int* CvCascadeBoostTrainData::get_cat_var_data( CvDTreeNode* n, int vi, int* catValuesBuf )
{
    int nodeSampleCount = n->sample_count;
    int* sampleIndicesBuf = catValuesBuf; //
    const int* sampleIndices = get_sample_indices(n, sampleIndicesBuf);

    if ( vi < numPrecalcVal )
    {
        for( int i = 0; i < nodeSampleCount; i++ )
            catValuesBuf[i] = (int) valCache.at<float>( vi, sampleIndices[i]);
    }
    else
    {
        if( vi >= numPrecalcVal && vi < var_count )
        {
            for( int i = 0; i < nodeSampleCount; i++ )
                catValuesBuf[i] = (int)(*featureEvaluator)( vi, sampleIndices[i] );
        }
        else
        {
            get_cv_labels( n, catValuesBuf );
        }
    }

    return catValuesBuf;
}

float CvCascadeBoostTrainData::getVarValue( int vi, int si )
{
    if ( vi < numPrecalcVal && !valCache.empty() )
        return valCache.at<float>( vi, si );
    return (*featureEvaluator)( vi, si );
}


struct FeatureIdxOnlyPrecalc : cv::ParallelLoopBody
{
    FeatureIdxOnlyPrecalc( const CvFeatureEvaluator* _featureEvaluator, CvMat* _buf, int _sample_count, bool _is_buf_16u )
    {
        featureEvaluator = _featureEvaluator;
        sample_count = _sample_count;
        udst = (unsigned short*)_buf->data.s;
        idst = _buf->data.i;
        is_buf_16u = _is_buf_16u;
    }
    void operator()( const cv::Range& range ) const
    {
        cv::AutoBuffer<float> valCache(sample_count);
        float* valCachePtr = valCache.data();
        for ( int fi = range.start; fi < range.end; fi++)
        {
            for( int si = 0; si < sample_count; si++ )
            {
                valCachePtr[si] = (*featureEvaluator)( fi, si );
                if ( is_buf_16u )
                    *(udst + (size_t)fi*sample_count + si) = (unsigned short)si;
                else
                    *(idst + (size_t)fi*sample_count + si) = si;
            }
            if ( is_buf_16u )
                std::sort(udst + (size_t)fi*sample_count, udst + (size_t)(fi + 1)*sample_count, LessThanIdx<float, unsigned short>(valCachePtr) );
            else
                std::sort(idst + (size_t)fi*sample_count, idst + (size_t)(fi + 1)*sample_count, LessThanIdx<float, int>(valCachePtr) );
        }
    }
    const CvFeatureEvaluator* featureEvaluator;
    int sample_count;
    int* idst;
    unsigned short* udst;
    bool is_buf_16u;
};

struct FeatureValAndIdxPrecalc : cv::ParallelLoopBody
{
    FeatureValAndIdxPrecalc( const CvFeatureEvaluator* _featureEvaluator, CvMat* _buf, cv::Mat* _valCache, int _sample_count, bool _is_buf_16u )
    {
        featureEvaluator = _featureEvaluator;
        valCache = _valCache;
        sample_count = _sample_count;
        udst = (unsigned short*)_buf->data.s;
        idst = _buf->data.i;
        is_buf_16u = _is_buf_16u;
    }
    void operator()( const cv::Range& range ) const
    {
        for ( int fi = range.start; fi < range.end; fi++)
        {
            for( int si = 0; si < sample_count; si++ )
            {
                valCache->at<float>(fi,si) = (*featureEvaluator)( fi, si );
                if ( is_buf_16u )
                    *(udst + (size_t)fi*sample_count + si) = (unsigned short)si;
                else
                    *(idst + (size_t)fi*sample_count + si) = si;
            }
            if ( is_buf_16u )
                std::sort(udst + (size_t)fi*sample_count, udst + (size_t)(fi + 1)*sample_count, LessThanIdx<float, unsigned short>(valCache->ptr<float>(fi)) );
            else
                std::sort(idst + (size_t)fi*sample_count, idst + (size_t)(fi + 1)*sample_count, LessThanIdx<float, int>(valCache->ptr<float>(fi)) );
        }
    }
    const CvFeatureEvaluator* featureEvaluator;
    cv::Mat* valCache;
    int sample_count;
    int* idst;
    unsigned short* udst;
    bool is_buf_16u;
};

struct FeatureValOnlyPrecalc : cv::ParallelLoopBody
{
    FeatureValOnlyPrecalc( const CvFeatureEvaluator* _featureEvaluator, cv::Mat* _valCache, int _sample_count )
    {
        featureEvaluator = _featureEvaluator;
        valCache = _valCache;
        sample_count = _sample_count;
    }
    void operator()( const cv::Range& range ) const
    {
        for ( int fi = range.start; fi < range.end; fi++)
            for( int si = 0; si < sample_count; si++ )
                valCache->at<float>(fi,si) = (*featureEvaluator)( fi, si );
    }
    const CvFeatureEvaluator* featureEvaluator;
    cv::Mat* valCache;
    int sample_count;
};

void CvCascadeBoostTrainData::precalculate()
{
    int minNum = MIN( numPrecalcVal, numPrecalcIdx);

    double proctime = -TIME( 0 );
    parallel_for_( cv::Range(numPrecalcVal, numPrecalcIdx),
                   FeatureIdxOnlyPrecalc(featureEvaluator, buf, sample_count, is_buf_16u!=0) );
    parallel_for_( cv::Range(0, minNum),
                   FeatureValAndIdxPrecalc(featureEvaluator, buf, &valCache, sample_count, is_buf_16u!=0) );
    parallel_for_( cv::Range(minNum, numPrecalcVal),
                   FeatureValOnlyPrecalc(featureEvaluator, &valCache, sample_count) );
    std::cout << "Precalculation time: " << (proctime + TIME( 0 )) << std::endl;
}