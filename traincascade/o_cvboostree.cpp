#include "o_cvboostree.h"

#include "o_utils.h"


static inline double
log_ratio( double val )
{
    const double eps = 1e-5;

    val = MAX( val, eps );
    val = MIN( val, 1. - eps );
    return log( val/(1. - val) );
}

CvBoostTree::CvBoostTree()
{
    ensemble = 0;
}


CvBoostTree::~CvBoostTree()
{
    clear();
}


void
CvBoostTree::clear()
{
    CvDTree::clear();
    ensemble = 0;
}


bool
CvBoostTree::train( CvDTreeTrainData* _train_data,
                    const CvMat* _subsample_idx, CvBoost* _ensemble )
{
    clear();
    ensemble = _ensemble;
    data = _train_data;
    data->shared = true;
    return do_train( _subsample_idx );
}


bool
CvBoostTree::train( const CvMat*, int, const CvMat*, const CvMat*,
                    const CvMat*, const CvMat*, const CvMat*, CvDTreeParams )
{
    assert(0);
    return false;
}


bool
CvBoostTree::train( CvDTreeTrainData*, const CvMat* )
{
    assert(0);
    return false;
}


void
CvBoostTree::scale( double _scale )
{
    CvDTreeNode* node = root;

    // traverse the tree and scale all the node values
    for(;;)
    {
        CvDTreeNode* parent;
        for(;;)
        {
            node->value *= _scale;
            if( !node->left )
                break;
            node = node->left;
        }

        for( parent = node->parent; parent && parent->right == node;
            node = parent, parent = parent->parent )
            ;

        if( !parent )
            break;

        node = parent->right;
    }
}


void
CvBoostTree::try_split_node( CvDTreeNode* node )
{
    CvDTree::try_split_node( node );

    if( !node->left )
    {
        // if the node has not been split,
        // store the responses for the corresponding training samples
        double* weak_eval = ensemble->get_weak_response()->data.db;
        cv::AutoBuffer<int> inn_buf(node->sample_count);
        const int* labels = data->get_cv_labels(node, inn_buf.data());
        int i, count = node->sample_count;
        double value = node->value;

        for( i = 0; i < count; i++ )
            weak_eval[labels[i]] = value;
    }
}


double
CvBoostTree::calc_node_dir( CvDTreeNode* node )
{
    char* dir = (char*)data->direction->data.ptr;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int i, n = node->sample_count, vi = node->split->var_idx;
    double L, R;

    assert( !node->split->inversed );

    if( data->get_var_type(vi) >= 0 ) // split on categorical var
    {
        cv::AutoBuffer<int> inn_buf(n);
        const int* cat_labels = data->get_cat_var_data(node, vi, inn_buf.data());
        const int* subset = node->split->subset;
        double sum = 0, sum_abs = 0;

        for( i = 0; i < n; i++ )
        {
            int idx = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
            double w = weights[i];
            int d = idx >= 0 ? CV_DTREE_CAT_DIR(idx,subset) : 0;
            sum += d*w; sum_abs += (d & 1)*w;
            dir[i] = (char)d;
        }

        R = (sum_abs + sum) * 0.5;
        L = (sum_abs - sum) * 0.5;
    }
    else // split on ordered var
    {
        cv::AutoBuffer<uchar> inn_buf(2*n*sizeof(int)+n*sizeof(float));
        float* values_buf = (float*)inn_buf.data();
        int* sorted_indices_buf = (int*)(values_buf + n);
        int* sample_indices_buf = sorted_indices_buf + n;
        const float* values = 0;
        const int* sorted_indices = 0;
        data->get_ord_var_data( node, vi, values_buf, sorted_indices_buf, &values, &sorted_indices, sample_indices_buf );
        int split_point = node->split->ord.split_point;
        int n1 = node->get_num_valid(vi);

        assert( 0 <= split_point && split_point < n1-1 );
        L = R = 0;

        for( i = 0; i <= split_point; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx];
            dir[idx] = (char)-1;
            L += w;
        }

        for( ; i < n1; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx];
            dir[idx] = (char)1;
            R += w;
        }

        for( ; i < n; i++ )
            dir[sorted_indices[i]] = (char)0;
    }

    node->maxlr = MAX( L, R );
    return node->split->quality/(L + R);
}


CvDTreeSplit*
CvBoostTree::find_split_ord_class( CvDTreeNode* node, int vi, float init_quality,
                                    CvDTreeSplit* _split, uchar* _ext_buf )
{
    const float epsilon = FLT_EPSILON*2;

    const double* weights = ensemble->get_subtree_weights()->data.db;
    int n = node->sample_count;
    int n1 = node->get_num_valid(vi);

    cv::AutoBuffer<uchar> inn_buf;
    if( !_ext_buf )
        inn_buf.allocate(n*(3*sizeof(int)+sizeof(float)));
    uchar* ext_buf = _ext_buf ? _ext_buf : inn_buf.data();
    float* values_buf = (float*)ext_buf;
    int* sorted_indices_buf = (int*)(values_buf + n);
    int* sample_indices_buf = sorted_indices_buf + n;
    const float* values = 0;
    const int* sorted_indices = 0;
    data->get_ord_var_data( node, vi, values_buf, sorted_indices_buf, &values, &sorted_indices, sample_indices_buf );
    int* responses_buf = sorted_indices_buf + n;
    const int* responses = data->get_class_labels( node, responses_buf );
    const double* rcw0 = weights + n;
    double lcw[2] = {0,0}, rcw[2];
    int i, best_i = -1;
    double best_val = init_quality;
    int boost_type = ensemble->get_params().boost_type;
    int split_criteria = ensemble->get_params().split_criteria;

    rcw[0] = rcw0[0]; rcw[1] = rcw0[1];
    for( i = n1; i < n; i++ )
    {
        int idx = sorted_indices[i];
        double w = weights[idx];
        rcw[responses[idx]] -= w;
    }

    if( split_criteria != CvBoost::GINI && split_criteria != CvBoost::MISCLASS )
        split_criteria = boost_type == CvBoost::DISCRETE ? CvBoost::MISCLASS : CvBoost::GINI;

    if( split_criteria == CvBoost::GINI )
    {
        double L = 0, R = rcw[0] + rcw[1];
        double lsum2 = 0, rsum2 = rcw[0]*rcw[0] + rcw[1]*rcw[1];

        for( i = 0; i < n1 - 1; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx], w2 = w*w;
            double lv, rv;
            idx = responses[idx];
            L += w; R -= w;
            lv = lcw[idx]; rv = rcw[idx];
            lsum2 += 2*lv*w + w2;
            rsum2 -= 2*rv*w - w2;
            lcw[idx] = lv + w; rcw[idx] = rv - w;

            if( values[i] + epsilon < values[i+1] )
            {
                double val = (lsum2*R + rsum2*L)/(L*R);
                if( best_val < val )
                {
                    best_val = val;
                    best_i = i;
                }
            }
        }
    }
    else
    {
        for( i = 0; i < n1 - 1; i++ )
        {
            int idx = sorted_indices[i];
            double w = weights[idx];
            idx = responses[idx];
            lcw[idx] += w;
            rcw[idx] -= w;

            if( values[i] + epsilon < values[i+1] )
            {
                double val = lcw[0] + rcw[1], val2 = lcw[1] + rcw[0];
                val = MAX(val, val2);
                if( best_val < val )
                {
                    best_val = val;
                    best_i = i;
                }
            }
        }
    }

    CvDTreeSplit* split = 0;
    if( best_i >= 0 )
    {
        split = _split ? _split : data->new_split_ord( 0, 0.0f, 0, 0, 0.0f );
        split->var_idx = vi;
        split->ord.c = (values[best_i] + values[best_i+1])*0.5f;
        split->ord.split_point = best_i;
        split->inversed = 0;
        split->quality = (float)best_val;
    }
    return split;
}

CvDTreeSplit*
CvBoostTree::find_split_cat_class( CvDTreeNode* node, int vi, float init_quality, CvDTreeSplit* _split, uchar* _ext_buf )
{
    int ci = data->get_var_type(vi);
    int n = node->sample_count;
    int mi = data->cat_count->data.i[ci];

    int base_size = (2*mi+3)*sizeof(double) + mi*sizeof(double*);
    cv::AutoBuffer<uchar> inn_buf((2*mi+3)*sizeof(double) + mi*sizeof(double*));
    if( !_ext_buf)
        inn_buf.allocate( base_size + 2*n*sizeof(int) );
    uchar* base_buf = inn_buf.data();
    uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;

    int* cat_labels_buf = (int*)ext_buf;
    const int* cat_labels = data->get_cat_var_data(node, vi, cat_labels_buf);
    int* responses_buf = cat_labels_buf + n;
    const int* responses = data->get_class_labels(node, responses_buf);
    double lcw[2]={0,0}, rcw[2]={0,0};

    double* cjk = (double*)cv::alignPtr(base_buf,sizeof(double))+2;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    double** dbl_ptr = (double**)(cjk + 2*mi);
    int i, j, k, idx;
    double L = 0, R;
    double best_val = init_quality;
    int best_subset = -1, subset_i;
    int boost_type = ensemble->get_params().boost_type;
    int split_criteria = ensemble->get_params().split_criteria;

    // init array of counters:
    // c_{jk} - number of samples that have vi-th input variable = j and response = k.
    for( j = -1; j < mi; j++ )
        cjk[j*2] = cjk[j*2+1] = 0;

    for( i = 0; i < n; i++ )
    {
        double w = weights[i];
        j = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
        k = responses[i];
        cjk[j*2 + k] += w;
    }

    for( j = 0; j < mi; j++ )
    {
        rcw[0] += cjk[j*2];
        rcw[1] += cjk[j*2+1];
        dbl_ptr[j] = cjk + j*2 + 1;
    }

    R = rcw[0] + rcw[1];

    if( split_criteria != CvBoost::GINI && split_criteria != CvBoost::MISCLASS )
        split_criteria = boost_type == CvBoost::DISCRETE ? CvBoost::MISCLASS : CvBoost::GINI;

    // sort rows of c_jk by increasing c_j,1
    // (i.e. by the weight of samples in j-th category that belong to class 1)
    std::sort(dbl_ptr, dbl_ptr + mi, LessThanPtr<double>());

    for( subset_i = 0; subset_i < mi-1; subset_i++ )
    {
        idx = (int)(dbl_ptr[subset_i] - cjk)/2;
        const double* crow = cjk + idx*2;
        double w0 = crow[0], w1 = crow[1];
        double weight = w0 + w1;

        if( weight < FLT_EPSILON )
            continue;

        lcw[0] += w0; rcw[0] -= w0;
        lcw[1] += w1; rcw[1] -= w1;

        if( split_criteria == CvBoost::GINI )
        {
            double lsum2 = lcw[0]*lcw[0] + lcw[1]*lcw[1];
            double rsum2 = rcw[0]*rcw[0] + rcw[1]*rcw[1];

            L += weight;
            R -= weight;

            if( L > FLT_EPSILON && R > FLT_EPSILON )
            {
                double val = (lsum2*R + rsum2*L)/(L*R);
                if( best_val < val )
                {
                    best_val = val;
                    best_subset = subset_i;
                }
            }
        }
        else
        {
            double val = lcw[0] + rcw[1];
            double val2 = lcw[1] + rcw[0];

            val = MAX(val, val2);
            if( best_val < val )
            {
                best_val = val;
                best_subset = subset_i;
            }
        }
    }

    CvDTreeSplit* split = 0;
    if( best_subset >= 0 )
    {
        split = _split ? _split : data->new_split_cat( 0, -1.0f);
        split->var_idx = vi;
        split->quality = (float)best_val;
        memset( split->subset, 0, (data->max_c_count + 31)/32 * sizeof(int));
        for( i = 0; i <= best_subset; i++ )
        {
            idx = (int)(dbl_ptr[i] - cjk) >> 1;
            split->subset[idx >> 5] |= 1 << (idx & 31);
        }
    }
    return split;
}


CvDTreeSplit*
CvBoostTree::find_split_ord_reg( CvDTreeNode* node, int vi, float init_quality, CvDTreeSplit* _split, uchar* _ext_buf )
{
    const float epsilon = FLT_EPSILON*2;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int n = node->sample_count;
    int n1 = node->get_num_valid(vi);

    cv::AutoBuffer<uchar> inn_buf;
    if( !_ext_buf )
        inn_buf.allocate(2*n*(sizeof(int)+sizeof(float)));
    uchar* ext_buf = _ext_buf ? _ext_buf : inn_buf.data();

    float* values_buf = (float*)ext_buf;
    int* indices_buf = (int*)(values_buf + n);
    int* sample_indices_buf = indices_buf + n;
    const float* values = 0;
    const int* indices = 0;
    data->get_ord_var_data( node, vi, values_buf, indices_buf, &values, &indices, sample_indices_buf );
    float* responses_buf = (float*)(indices_buf + n);
    const float* responses = data->get_ord_responses( node, responses_buf, sample_indices_buf );

    int i, best_i = -1;
    double L = 0, R = weights[n];
    double best_val = init_quality, lsum = 0, rsum = node->value*R;

    // compensate for missing values
    for( i = n1; i < n; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        rsum -= responses[idx]*w;
        R -= w;
    }

    // find the optimal split
    for( i = 0; i < n1 - 1; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        double t = responses[idx]*w;
        L += w; R -= w;
        lsum += t; rsum -= t;

        if( values[i] + epsilon < values[i+1] )
        {
            double val = (lsum*lsum*R + rsum*rsum*L)/(L*R);
            if( best_val < val )
            {
                best_val = val;
                best_i = i;
            }
        }
    }

    CvDTreeSplit* split = 0;
    if( best_i >= 0 )
    {
        split = _split ? _split : data->new_split_ord( 0, 0.0f, 0, 0, 0.0f );
        split->var_idx = vi;
        split->ord.c = (values[best_i] + values[best_i+1])*0.5f;
        split->ord.split_point = best_i;
        split->inversed = 0;
        split->quality = (float)best_val;
    }
    return split;
}


CvDTreeSplit*
CvBoostTree::find_split_cat_reg( CvDTreeNode* node, int vi, float init_quality, CvDTreeSplit* _split, uchar* _ext_buf )
{
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int ci = data->get_var_type(vi);
    int n = node->sample_count;
    int mi = data->cat_count->data.i[ci];
    int base_size = (2*mi+3)*sizeof(double) + mi*sizeof(double*);
    cv::AutoBuffer<uchar> inn_buf(base_size);
    if( !_ext_buf )
        inn_buf.allocate(base_size + n*(2*sizeof(int) + sizeof(float)));
    uchar* base_buf = inn_buf.data();
    uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;

    int* cat_labels_buf = (int*)ext_buf;
    const int* cat_labels = data->get_cat_var_data(node, vi, cat_labels_buf);
    float* responses_buf = (float*)(cat_labels_buf + n);
    int* sample_indices_buf = (int*)(responses_buf + n);
    const float* responses = data->get_ord_responses(node, responses_buf, sample_indices_buf);

    double* sum = (double*)cv::alignPtr(base_buf,sizeof(double)) + 1;
    double* counts = sum + mi + 1;
    double** sum_ptr = (double**)(counts + mi);
    double L = 0, R = 0, best_val = init_quality, lsum = 0, rsum = 0;
    int i, best_subset = -1, subset_i;

    for( i = -1; i < mi; i++ )
        sum[i] = counts[i] = 0;

    // calculate sum response and weight of each category of the input var
    for( i = 0; i < n; i++ )
    {
        int idx = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
        double w = weights[i];
        double s = sum[idx] + responses[i]*w;
        double nc = counts[idx] + w;
        sum[idx] = s;
        counts[idx] = nc;
    }

    // calculate average response in each category
    for( i = 0; i < mi; i++ )
    {
        R += counts[i];
        rsum += sum[i];
        sum[i] = fabs(counts[i]) > DBL_EPSILON ? sum[i]/counts[i] : 0;
        sum_ptr[i] = sum + i;
    }

    std::sort(sum_ptr, sum_ptr + mi, LessThanPtr<double>());

    // revert back to unnormalized sums
    // (there should be a very little loss in accuracy)
    for( i = 0; i < mi; i++ )
        sum[i] *= counts[i];

    for( subset_i = 0; subset_i < mi-1; subset_i++ )
    {
        int idx = (int)(sum_ptr[subset_i] - sum);
        double ni = counts[idx];

        if( ni > FLT_EPSILON )
        {
            double s = sum[idx];
            lsum += s; L += ni;
            rsum -= s; R -= ni;

            if( L > FLT_EPSILON && R > FLT_EPSILON )
            {
                double val = (lsum*lsum*R + rsum*rsum*L)/(L*R);
                if( best_val < val )
                {
                    best_val = val;
                    best_subset = subset_i;
                }
            }
        }
    }

    CvDTreeSplit* split = 0;
    if( best_subset >= 0 )
    {
        split = _split ? _split : data->new_split_cat( 0, -1.0f);
        split->var_idx = vi;
        split->quality = (float)best_val;
        memset( split->subset, 0, (data->max_c_count + 31)/32 * sizeof(int));
        for( i = 0; i <= best_subset; i++ )
        {
            int idx = (int)(sum_ptr[i] - sum);
            split->subset[idx >> 5] |= 1 << (idx & 31);
        }
    }
    return split;
}


CvDTreeSplit*
CvBoostTree::find_surrogate_split_ord( CvDTreeNode* node, int vi, uchar* _ext_buf )
{
    const float epsilon = FLT_EPSILON*2;
    int n = node->sample_count;
    cv::AutoBuffer<uchar> inn_buf;
    if( !_ext_buf )
        inn_buf.allocate(n*(2*sizeof(int)+sizeof(float)));
    uchar* ext_buf = _ext_buf ? _ext_buf : inn_buf.data();
    float* values_buf = (float*)ext_buf;
    int* indices_buf = (int*)(values_buf + n);
    int* sample_indices_buf = indices_buf + n;
    const float* values = 0;
    const int* indices = 0;
    data->get_ord_var_data( node, vi, values_buf, indices_buf, &values, &indices, sample_indices_buf );

    const double* weights = ensemble->get_subtree_weights()->data.db;
    const char* dir = (char*)data->direction->data.ptr;
    int n1 = node->get_num_valid(vi);
    // LL - number of samples that both the primary and the surrogate splits send to the left
    // LR - ... primary split sends to the left and the surrogate split sends to the right
    // RL - ... primary split sends to the right and the surrogate split sends to the left
    // RR - ... both send to the right
    int i, best_i = -1, best_inversed = 0;
    double best_val;
    double LL = 0, RL = 0, LR, RR;
    double worst_val = node->maxlr;
    double sum = 0, sum_abs = 0;
    best_val = worst_val;

    for( i = 0; i < n1; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        int d = dir[idx];
        sum += d*w; sum_abs += (d & 1)*w;
    }

    // sum_abs = R + L; sum = R - L
    RR = (sum_abs + sum)*0.5;
    LR = (sum_abs - sum)*0.5;

    // initially all the samples are sent to the right by the surrogate split,
    // LR of them are sent to the left by primary split, and RR - to the right.
    // now iteratively compute LL, LR, RL and RR for every possible surrogate split value.
    for( i = 0; i < n1 - 1; i++ )
    {
        int idx = indices[i];
        double w = weights[idx];
        int d = dir[idx];

        if( d < 0 )
        {
            LL += w; LR -= w;
            if( LL + RR > best_val && values[i] + epsilon < values[i+1] )
            {
                best_val = LL + RR;
                best_i = i; best_inversed = 0;
            }
        }
        else if( d > 0 )
        {
            RL += w; RR -= w;
            if( RL + LR > best_val && values[i] + epsilon < values[i+1] )
            {
                best_val = RL + LR;
                best_i = i; best_inversed = 1;
            }
        }
    }

    return best_i >= 0 && best_val > node->maxlr ? data->new_split_ord( vi,
        (values[best_i] + values[best_i+1])*0.5f, best_i,
        best_inversed, (float)best_val ) : 0;
}


CvDTreeSplit*
CvBoostTree::find_surrogate_split_cat( CvDTreeNode* node, int vi, uchar* _ext_buf )
{
    const char* dir = (char*)data->direction->data.ptr;
    const double* weights = ensemble->get_subtree_weights()->data.db;
    int n = node->sample_count;
    int i, mi = data->cat_count->data.i[data->get_var_type(vi)];

    int base_size = (2*mi+3)*sizeof(double);
    cv::AutoBuffer<uchar> inn_buf(base_size);
    if( !_ext_buf )
        inn_buf.allocate(base_size + n*sizeof(int));
    uchar* ext_buf = _ext_buf ? _ext_buf : inn_buf.data();
    int* cat_labels_buf = (int*)ext_buf;
    const int* cat_labels = data->get_cat_var_data(node, vi, cat_labels_buf);

    // LL - number of samples that both the primary and the surrogate splits send to the left
    // LR - ... primary split sends to the left and the surrogate split sends to the right
    // RL - ... primary split sends to the right and the surrogate split sends to the left
    // RR - ... both send to the right
    CvDTreeSplit* split = data->new_split_cat( vi, 0 );
    double best_val = 0;
    double* lc = (double*)cv::alignPtr(cat_labels_buf + n, sizeof(double)) + 1;
    double* rc = lc + mi + 1;

    for( i = -1; i < mi; i++ )
        lc[i] = rc[i] = 0;

    // 1. for each category calculate the weight of samples
    // sent to the left (lc) and to the right (rc) by the primary split
    for( i = 0; i < n; i++ )
    {
        int idx = ((cat_labels[i] == 65535) && data->is_buf_16u) ? -1 : cat_labels[i];
        double w = weights[i];
        int d = dir[i];
        double sum = lc[idx] + d*w;
        double sum_abs = rc[idx] + (d & 1)*w;
        lc[idx] = sum; rc[idx] = sum_abs;
    }

    for( i = 0; i < mi; i++ )
    {
        double sum = lc[i];
        double sum_abs = rc[i];
        lc[i] = (sum_abs - sum) * 0.5;
        rc[i] = (sum_abs + sum) * 0.5;
    }

    // 2. now form the split.
    // in each category send all the samples to the same direction as majority
    for( i = 0; i < mi; i++ )
    {
        double lval = lc[i], rval = rc[i];
        if( lval > rval )
        {
            split->subset[i >> 5] |= 1 << (i & 31);
            best_val += lval;
        }
        else
            best_val += rval;
    }

    split->quality = (float)best_val;
    if( split->quality <= node->maxlr )
        cvSetRemoveByPtr( data->split_heap, split ), split = 0;

    return split;
}


void
CvBoostTree::calc_node_value( CvDTreeNode* node )
{
    int i, n = node->sample_count;
    const double* weights = ensemble->get_weights()->data.db;
    cv::AutoBuffer<uchar> inn_buf(n*(sizeof(int) + ( data->is_classifier ? sizeof(int) : sizeof(int) + sizeof(float))));
    int* labels_buf = (int*)inn_buf.data();
    const int* labels = data->get_cv_labels(node, labels_buf);
    double* subtree_weights = ensemble->get_subtree_weights()->data.db;
    double rcw[2] = {0,0};
    int boost_type = ensemble->get_params().boost_type;

    if( data->is_classifier )
    {
        int* _responses_buf = labels_buf + n;
        const int* _responses = data->get_class_labels(node, _responses_buf);
        int m = data->get_num_classes();
        int* cls_count = data->counts->data.i;
        for( int k = 0; k < m; k++ )
            cls_count[k] = 0;

        for( i = 0; i < n; i++ )
        {
            int idx = labels[i];
            double w = weights[idx];
            int r = _responses[i];
            rcw[r] += w;
            cls_count[r]++;
            subtree_weights[i] = w;
        }

        node->class_idx = rcw[1] > rcw[0];

        if( boost_type == CvBoost::DISCRETE )
        {
            // ignore cat_map for responses, and use {-1,1},
            // as the whole ensemble response is computes as sign(sum_i(weak_response_i)
            node->value = node->class_idx*2 - 1;
        }
        else
        {
            double p = rcw[1]/(rcw[0] + rcw[1]);
            assert( boost_type == CvBoost::REAL );

            // store log-ratio of the probability
            node->value = 0.5*log_ratio(p);
        }
    }
    else
    {
        // in case of regression tree:
        //  * node value is 1/n*sum_i(Y_i), where Y_i is i-th response,
        //    n is the number of samples in the node.
        //  * node risk is the sum of squared errors: sum_i((Y_i - <node_value>)^2)
        double sum = 0, sum2 = 0, iw;
        float* values_buf = (float*)(labels_buf + n);
        int* sample_indices_buf = (int*)(values_buf + n);
        const float* values = data->get_ord_responses(node, values_buf, sample_indices_buf);

        for( i = 0; i < n; i++ )
        {
            int idx = labels[i];
            double w = weights[idx]/*priors[values[i] > 0]*/;
            double t = values[i];
            rcw[0] += w;
            subtree_weights[i] = w;
            sum += t*w;
            sum2 += t*t*w;
        }

        iw = 1./rcw[0];
        node->value = sum*iw;
        node->node_risk = sum2 - (sum*iw)*sum;

        // renormalize the risk, as in try_split_node the unweighted formula
        // sqrt(risk)/n is used, rather than sqrt(risk)/sum(weights_i)
        node->node_risk *= n*iw*n*iw;
    }

    // store summary weights
    subtree_weights[n] = rcw[0];
    subtree_weights[n+1] = rcw[1];
}
