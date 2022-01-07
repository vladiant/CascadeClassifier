
#include <queue>

#include "cascadeclassifier.h"

#include "o_cvcascadeboosttree.h"
#include "o_cvcascadeboosttraindata.h"
#include "o_cvdtreenode.h"

// TODO: Duplicated!
#define CV_DTREE_CAT_DIR(idx,subset) \
    (2*((subset[(idx)>>5]&(1 << ((idx) & 31)))==0)-1)

CvDTreeNode* CvCascadeBoostTree::predict( int sampleIdx ) const
{
    CvDTreeNode* node = root;
    if( !node )
        CV_Error( CV_StsError, "The tree has not been trained yet" );

    if ( ((CvCascadeBoostTrainData*)data)->featureEvaluator->getMaxCatCount() == 0 ) // ordered
    {
        while( node->left )
        {
            CvDTreeSplit* split = node->split;
            float val = ((CvCascadeBoostTrainData*)data)->getVarValue( split->var_idx, sampleIdx );
            node = val <= split->ord.c ? node->left : node->right;
        }
    }
    else // categorical
    {
        while( node->left )
        {
            CvDTreeSplit* split = node->split;
            int c = (int)((CvCascadeBoostTrainData*)data)->getVarValue( split->var_idx, sampleIdx );
            node = CV_DTREE_CAT_DIR(c, split->subset) < 0 ? node->left : node->right;
        }
    }
    return node;
}

void CvCascadeBoostTree::write(cv::FileStorage &fs, const cv::Mat& featureMap )
{
    int maxCatCount = ((CvCascadeBoostTrainData*)data)->featureEvaluator->getMaxCatCount();
    int subsetN = (maxCatCount + 31)/32;
    std::queue<CvDTreeNode*> internalNodesQueue;
    int size = (int)pow( 2.f, (float)ensemble->get_params().max_depth);
    std::vector<float> leafVals(size);
    int leafValIdx = 0;
    int internalNodeIdx = 1;
    CvDTreeNode* tempNode;

    CV_DbgAssert( root );
    internalNodesQueue.push( root );

    fs << "{";
    fs << CC_INTERNAL_NODES << "[:";
    while (!internalNodesQueue.empty())
    {
        tempNode = internalNodesQueue.front();
        CV_Assert( tempNode->left );
        if ( !tempNode->left->left && !tempNode->left->right) // left node is leaf
        {
            leafVals[-leafValIdx] = (float)tempNode->left->value;
            fs << leafValIdx-- ;
        }
        else
        {
            internalNodesQueue.push( tempNode->left );
            fs << internalNodeIdx++;
        }
        CV_Assert( tempNode->right );
        if ( !tempNode->right->left && !tempNode->right->right) // right node is leaf
        {
            leafVals[-leafValIdx] = (float)tempNode->right->value;
            fs << leafValIdx--;
        }
        else
        {
            internalNodesQueue.push( tempNode->right );
            fs << internalNodeIdx++;
        }
        int fidx = tempNode->split->var_idx;
        fidx = featureMap.empty() ? fidx : featureMap.at<int>(0, fidx);
        fs << fidx;
        if ( !maxCatCount )
            fs << tempNode->split->ord.c;
        else
            for( int i = 0; i < subsetN; i++ )
                fs << tempNode->split->subset[i];
        internalNodesQueue.pop();
    }
    fs << "]"; // CC_INTERNAL_NODES

    fs << CC_LEAF_VALUES << "[:";
    for (int ni = 0; ni < -leafValIdx; ni++)
        fs << leafVals[ni];
    fs << "]"; // CC_LEAF_VALUES
    fs << "}";
}

void CvCascadeBoostTree::read( const cv::FileNode &node, CvBoost* _ensemble,
                                CvDTreeTrainData* _data )
{
    int maxCatCount = ((CvCascadeBoostTrainData*)_data)->featureEvaluator->getMaxCatCount();
    int subsetN = (maxCatCount + 31)/32;
    int step = 3 + ( maxCatCount>0 ? subsetN : 1 );

    std::queue<CvDTreeNode*> internalNodesQueue;
    cv::FileNodeIterator internalNodesIt, leafValsuesIt;
    CvDTreeNode* prntNode, *cldNode;

    clear();
    data = _data;
    ensemble = _ensemble;
    pruned_tree_idx = 0;

    // read tree nodes
    cv::FileNode rnode = node[CC_INTERNAL_NODES];
    internalNodesIt = rnode.end();
    leafValsuesIt = node[CC_LEAF_VALUES].end();
    internalNodesIt++; leafValsuesIt++;
    for( size_t i = 0; i < rnode.size()/step; i++ )
    {
        prntNode = data->new_node( 0, 0, 0, 0 );
        if ( maxCatCount > 0 )
        {
            prntNode->split = data->new_split_cat( 0, 0 );
            for( int j = subsetN-1; j>=0; j--)
            {
                *internalNodesIt >> prntNode->split->subset[j]; internalNodesIt++;
            }
        }
        else
        {
            float split_value;
            *internalNodesIt >> split_value; internalNodesIt++;
            prntNode->split = data->new_split_ord( 0, split_value, 0, 0, 0);
        }
        *internalNodesIt >> prntNode->split->var_idx; internalNodesIt++;
        int ridx, lidx;
        *internalNodesIt >> ridx; internalNodesIt++;
        *internalNodesIt >> lidx;internalNodesIt++;
        if ( ridx <= 0)
        {
            prntNode->right = cldNode = data->new_node( 0, 0, 0, 0 );
            *leafValsuesIt >> cldNode->value; leafValsuesIt++;
            cldNode->parent = prntNode;
        }
        else
        {
            prntNode->right = internalNodesQueue.front();
            prntNode->right->parent = prntNode;
            internalNodesQueue.pop();
        }

        if ( lidx <= 0)
        {
            prntNode->left = cldNode = data->new_node( 0, 0, 0, 0 );
            *leafValsuesIt >> cldNode->value; leafValsuesIt++;
            cldNode->parent = prntNode;
        }
        else
        {
            prntNode->left = internalNodesQueue.front();
            prntNode->left->parent = prntNode;
            internalNodesQueue.pop();
        }

        internalNodesQueue.push( prntNode );
    }

    root = internalNodesQueue.front();
    internalNodesQueue.pop();
}

void CvCascadeBoostTree::split_node_data( CvDTreeNode* node )
{
    int n = node->sample_count, nl, nr, scount = data->sample_count;
    char* dir = (char*)data->direction->data.ptr;
    CvDTreeNode *left = 0, *right = 0;
    int* newIdx = data->split_buf->data.i;
    int newBufIdx = data->get_child_buf_idx( node );
    int workVarCount = data->get_work_var_count();
    CvMat* buf = data->buf;
    size_t length_buf_row = data->get_length_subbuf();
    cv::AutoBuffer<uchar> inn_buf(n*(3*sizeof(int)+sizeof(float)));
    int* tempBuf = (int*)inn_buf.data();
    bool splitInputData;

    complete_node_dir(node);

    for( int i = nl = nr = 0; i < n; i++ )
    {
        int d = dir[i];
        // initialize new indices for splitting ordered variables
        newIdx[i] = (nl & (d-1)) | (nr & -d); // d ? ri : li
        nr += d;
        nl += d^1;
    }

    node->left = left = data->new_node( node, nl, newBufIdx, node->offset );
    node->right = right = data->new_node( node, nr, newBufIdx, node->offset + nl );

    splitInputData = node->depth + 1 < data->params.max_depth &&
        (node->left->sample_count > data->params.min_sample_count ||
        node->right->sample_count > data->params.min_sample_count);

    // split ordered variables, keep both halves sorted.
    for( int vi = 0; vi < ((CvCascadeBoostTrainData*)data)->numPrecalcIdx; vi++ )
    {
        int ci = data->get_var_type(vi);
        if( ci >= 0 || !splitInputData )
            continue;

        int n1 = node->get_num_valid(vi);
        float *src_val_buf = (float*)(tempBuf + n);
        int *src_sorted_idx_buf = (int*)(src_val_buf + n);
        int *src_sample_idx_buf = src_sorted_idx_buf + n;
        const int* src_sorted_idx = 0;
        const float* src_val = 0;
        data->get_ord_var_data(node, vi, src_val_buf, src_sorted_idx_buf, &src_val, &src_sorted_idx, src_sample_idx_buf);

        for(int i = 0; i < n; i++)
            tempBuf[i] = src_sorted_idx[i];

        if (data->is_buf_16u)
        {
            ushort *ldst, *rdst;
            ldst = (ushort*)(buf->data.s + left->buf_idx*length_buf_row +
                vi*scount + left->offset);
            rdst = (ushort*)(ldst + nl);

            // split sorted
            for( int i = 0; i < n1; i++ )
            {
                int idx = tempBuf[i];
                int d = dir[idx];
                idx = newIdx[idx];
                if (d)
                {
                    *rdst = (ushort)idx;
                    rdst++;
                }
                else
                {
                    *ldst = (ushort)idx;
                    ldst++;
                }
            }
            CV_Assert( n1 == n );
        }
        else
        {
            int *ldst, *rdst;
            ldst = buf->data.i + left->buf_idx*length_buf_row +
                vi*scount + left->offset;
            rdst = buf->data.i + right->buf_idx*length_buf_row +
                vi*scount + right->offset;

            // split sorted
            for( int i = 0; i < n1; i++ )
            {
                int idx = tempBuf[i];
                int d = dir[idx];
                idx = newIdx[idx];
                if (d)
                {
                    *rdst = idx;
                    rdst++;
                }
                else
                {
                    *ldst = idx;
                    ldst++;
                }
            }
            CV_Assert( n1 == n );
        }
    }

    // split cv_labels using newIdx relocation table
    int *src_lbls_buf = tempBuf + n;
    const int* src_lbls = data->get_cv_labels(node, src_lbls_buf);

    for(int i = 0; i < n; i++)
        tempBuf[i] = src_lbls[i];

    if (data->is_buf_16u)
    {
        unsigned short *ldst = (unsigned short *)(buf->data.s + left->buf_idx*length_buf_row +
            (size_t)(workVarCount-1)*scount + left->offset);
        unsigned short *rdst = (unsigned short *)(buf->data.s + right->buf_idx*length_buf_row +
            (size_t)(workVarCount-1)*scount + right->offset);

        for( int i = 0; i < n; i++ )
        {
            int idx = tempBuf[i];
            if (dir[i])
            {
                *rdst = (unsigned short)idx;
                rdst++;
            }
            else
            {
                *ldst = (unsigned short)idx;
                ldst++;
            }
        }

    }
    else
    {
        int *ldst = buf->data.i + left->buf_idx*length_buf_row +
            (size_t)(workVarCount-1)*scount + left->offset;
        int *rdst = buf->data.i + right->buf_idx*length_buf_row +
            (size_t)(workVarCount-1)*scount + right->offset;

        for( int i = 0; i < n; i++ )
        {
            int idx = tempBuf[i];
            if (dir[i])
            {
                *rdst = idx;
                rdst++;
            }
            else
            {
                *ldst = idx;
                ldst++;
            }
        }
    }

    // split sample indices
    int *sampleIdx_src_buf = tempBuf + n;
    const int* sampleIdx_src = data->get_sample_indices(node, sampleIdx_src_buf);

    for(int i = 0; i < n; i++)
        tempBuf[i] = sampleIdx_src[i];

    if (data->is_buf_16u)
    {
        unsigned short* ldst = (unsigned short*)(buf->data.s + left->buf_idx*length_buf_row +
            (size_t)workVarCount*scount + left->offset);
        unsigned short* rdst = (unsigned short*)(buf->data.s + right->buf_idx*length_buf_row +
            (size_t)workVarCount*scount + right->offset);
        for (int i = 0; i < n; i++)
        {
            unsigned short idx = (unsigned short)tempBuf[i];
            if (dir[i])
            {
                *rdst = idx;
                rdst++;
            }
            else
            {
                *ldst = idx;
                ldst++;
            }
        }
    }
    else
    {
        int* ldst = buf->data.i + left->buf_idx*length_buf_row +
            (size_t)workVarCount*scount + left->offset;
        int* rdst = buf->data.i + right->buf_idx*length_buf_row +
            (size_t)workVarCount*scount + right->offset;
        for (int i = 0; i < n; i++)
        {
            int idx = tempBuf[i];
            if (dir[i])
            {
                *rdst = idx;
                rdst++;
            }
            else
            {
                *ldst = idx;
                ldst++;
            }
        }
    }

    for( int vi = 0; vi < data->var_count; vi++ )
    {
        left->set_num_valid(vi, (int)(nl));
        right->set_num_valid(vi, (int)(nr));
    }

    // deallocate the parent node data that is not needed anymore
    data->free_node_data(node);
}

static void auxMarkFeaturesInMap( const CvDTreeNode* node, cv::Mat& featureMap)
{
    if ( node && node->split )
    {
        featureMap.ptr<int>(0)[node->split->var_idx] = 1;
        auxMarkFeaturesInMap( node->left, featureMap );
        auxMarkFeaturesInMap( node->right, featureMap );
    }
}

void CvCascadeBoostTree::markFeaturesInMap( cv::Mat& featureMap )
{
    auxMarkFeaturesInMap( root, featureMap );
}