#pragma once

#include "o_cvdtree.h"
#include "o_cvboost.h"

class CvBoostTree: public CvDTree
{
public:
    CvBoostTree();
    virtual ~CvBoostTree();

    virtual bool train( CvDTreeTrainData* trainData,
                        const CvMat* subsample_idx, CvBoost* ensemble );

    virtual void scale( double s );
    // virtual void read( CvFileStorage* fs, CvFileNode* node,
    //                    CvBoost* ensemble, CvDTreeTrainData* _data );
    virtual void clear();

    /* dummy methods to avoid warnings: BEGIN */
    virtual bool train( const CvMat* trainData, int tflag,
                        const CvMat* responses, const CvMat* varIdx=0,
                        const CvMat* sampleIdx=0, const CvMat* varType=0,
                        const CvMat* missingDataMask=0,
                        CvDTreeParams params=CvDTreeParams() );
    virtual bool train( CvDTreeTrainData* trainData, const CvMat* _subsample_idx );

    // virtual void read( CvFileStorage* fs, CvFileNode* node );
    // virtual void read( CvFileStorage* fs, CvFileNode* node,
    //                    CvDTreeTrainData* data );
    /* dummy methods to avoid warnings: END */

protected:

    virtual void try_split_node( CvDTreeNode* n );
    virtual CvDTreeSplit* find_surrogate_split_ord( CvDTreeNode* n, int vi, uchar* ext_buf = 0 );
    virtual CvDTreeSplit* find_surrogate_split_cat( CvDTreeNode* n, int vi, uchar* ext_buf = 0 );
    virtual CvDTreeSplit* find_split_ord_class( CvDTreeNode* n, int vi,
        float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0 );
    virtual CvDTreeSplit* find_split_cat_class( CvDTreeNode* n, int vi,
        float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0 );
    virtual CvDTreeSplit* find_split_ord_reg( CvDTreeNode* n, int vi,
        float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0 );
    virtual CvDTreeSplit* find_split_cat_reg( CvDTreeNode* n, int vi,
        float init_quality = 0, CvDTreeSplit* _split = 0, uchar* ext_buf = 0 );
    virtual void calc_node_value( CvDTreeNode* n );
    virtual double calc_node_dir( CvDTreeNode* n );

    CvBoost* ensemble;
};