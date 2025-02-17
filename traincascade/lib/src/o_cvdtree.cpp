#include "o_cvdtree.h"

#include <cmath>

#include "o_blockedrange.h"
#include "o_cvdtreenode.h"
#include "o_cvdtreeparams.h"
#include "o_cvdtreesplit.h"
#include "o_cvdtreetraindata.h"
#include "o_utils.h"

CvDTree::CvDTree() {
  data = nullptr;
  var_importance = nullptr;
  default_model_name = "my_tree";

  clear();
}

void CvDTree::clear() {
  cvReleaseMat(&var_importance);
  if (data) {
    if (!data->shared)
      delete data;
    else
      free_tree();
    data = nullptr;
  }
  root = nullptr;
  pruned_tree_idx = -1;
}

CvDTree::~CvDTree() { clear(); }

const CvDTreeNode* CvDTree::get_root() const { return root; }

CvDTreeTrainData* CvDTree::get_data() { return data; }

bool CvDTree::train(const CvMat* _train_data, int _tflag,
                    const CvMat* _responses, const CvMat* _var_idx,
                    const CvMat* _sample_idx, const CvMat* _var_type,
                    const CvMat* _missing_mask, CvDTreeParams _params) {
  bool result = false;

  CV_FUNCNAME("CvDTree::train");

  __CV_BEGIN__;

  clear();
  data = new CvDTreeTrainData(_train_data, _tflag, _responses, _var_idx,
                              _sample_idx, _var_type, _missing_mask, _params,
                              false);
  CV_CALL(result = do_train(nullptr));

  __CV_END__;

  return result;
}

bool CvDTree::train(const cv::Mat& _train_data, int _tflag,
                    const cv::Mat& _responses, const cv::Mat& _var_idx,
                    const cv::Mat& _sample_idx, const cv::Mat& _var_type,
                    const cv::Mat& _missing_mask, CvDTreeParams _params) {
  train_data_hdr = cvMat(_train_data);
  train_data_mat = _train_data;
  responses_hdr = cvMat(_responses);
  responses_mat = _responses;

  CvMat vidx = cvMat(_var_idx), sidx = cvMat(_sample_idx),
        vtype = cvMat(_var_type), mmask = cvMat(_missing_mask);

  return train(&train_data_hdr, _tflag, &responses_hdr,
               vidx.data.ptr ? &vidx : nullptr, sidx.data.ptr ? &sidx : nullptr,
               vtype.data.ptr ? &vtype : nullptr, mmask.data.ptr ? &mmask : nullptr,
               _params);
}

bool CvDTree::train(CvDTreeTrainData* _data, const CvMat* _subsample_idx) {
  bool result = false;

  CV_FUNCNAME("CvDTree::train");

  __CV_BEGIN__;

  clear();
  data = _data;
  data->shared = true;
  CV_CALL(result = do_train(_subsample_idx));

  __CV_END__;

  return result;
}

bool CvDTree::do_train(const CvMat* _subsample_idx) {
  bool result = false;

  CV_FUNCNAME("CvDTree::do_train");

  __CV_BEGIN__;

  root = data->subsample_data(_subsample_idx);

  CV_CALL(try_split_node(root));

  if (root->split) {
    CV_Assert(root->left);
    CV_Assert(root->right);

    if (data->params.cv_folds > 0) CV_CALL(prune_cv());

    if (!data->shared) data->free_train_data();

    result = true;
  }

  __CV_END__;

  return result;
}

void CvDTree::try_split_node(CvDTreeNode* node) {
  CvDTreeSplit* best_split = nullptr;
  int i = 0, n = node->sample_count, vi = 0;
  bool can_split = true;
  double quality_scale = NAN;

  calc_node_value(node);

  if (node->sample_count <= data->params.min_sample_count ||
      node->depth >= data->params.max_depth)
    can_split = false;

  if (can_split && data->is_classifier) {
    // check if we have a "pure" node,
    // we assume that cls_count is filled by calc_node_value()
    int* cls_count = data->counts->data.i;
    int nz = 0, m = data->get_num_classes();
    for (i = 0; i < m; i++) nz += cls_count[i] != 0;
    if (nz == 1)  // there is only one class
      can_split = false;
  } else if (can_split) {
    if (sqrt(node->node_risk) / n < data->params.regression_accuracy)
      can_split = false;
  }

  if (can_split) {
    best_split = find_best_split(node);
    // TODO: check the split quality ...
    node->split = best_split;
  }
  if (!can_split || !best_split) {
    data->free_node_data(node);
    return;
  }

  quality_scale = calc_node_dir(node);
  if (data->params.use_surrogates) {
    // find all the surrogate splits
    // and sort them by their similarity to the primary one
    for (vi = 0; vi < data->var_count; vi++) {
      CvDTreeSplit* split = nullptr;
      int ci = data->get_var_type(vi);

      if (vi == best_split->var_idx) continue;

      if (ci >= 0)
        split = find_surrogate_split_cat(node, vi);
      else
        split = find_surrogate_split_ord(node, vi);

      if (split) {
        // insert the split
        CvDTreeSplit* prev_split = node->split;
        split->quality = (float)(split->quality * quality_scale);

        while (prev_split->next && prev_split->next->quality > split->quality)
          prev_split = prev_split->next;
        split->next = prev_split->next;
        prev_split->next = split;
      }
    }
  }
  split_node_data(node);
  try_split_node(node->left);
  try_split_node(node->right);
}

// calculate direction (left(-1),right(1),missing(0))
// for each sample using the best split
// the function returns scale coefficients for surrogate split quality factors.
// the scale is applied to normalize surrogate split quality relatively to the
// best (primary) split quality. That is, if a surrogate split is absolutely
// identical to the primary split, its quality will be set to the maximum value
// = quality of the primary split; otherwise, it will be lower. besides, the
// function compute node->maxlr, minimum possible quality (w/o considering the
// above mentioned scale) for a surrogate split. Surrogate splits with quality
// less than node->maxlr are not discarded.
double CvDTree::calc_node_dir(CvDTreeNode* node) {
  char* dir = (char*)data->direction->data.ptr;
  int i = 0, n = node->sample_count, vi = node->split->var_idx;
  double L = NAN, R = NAN;

  assert(!node->split->inversed);

  if (data->get_var_type(vi) >= 0)  // split on categorical var
  {
    cv::AutoBuffer<int> inn_buf((size_t)n * (!data->have_priors ? 1 : 2));
    int* labels_buf = inn_buf.data();
    const int* labels = data->get_cat_var_data(node, vi, labels_buf);
    const int* subset = node->split->subset;
    if (!data->have_priors) {
      int sum = 0, sum_abs = 0;

      for (i = 0; i < n; i++) {
        int idx = labels[i];
        int d = (((idx >= 0) && (!data->is_buf_16u)) ||
                 ((idx != 65535) && (data->is_buf_16u)))
                    ? CV_DTREE_CAT_DIR(idx, subset)
                    : 0;
        sum += d;
        sum_abs += d & 1;
        dir[i] = (char)d;
      }

      R = (sum_abs + sum) >> 1;
      L = (sum_abs - sum) >> 1;
    } else {
      const double* priors = data->priors_mult->data.db;
      double sum = 0, sum_abs = 0;
      int* responses_buf = labels_buf + n;
      const int* responses = data->get_class_labels(node, responses_buf);

      for (i = 0; i < n; i++) {
        int idx = labels[i];
        double w = priors[responses[i]];
        int d = idx >= 0 ? CV_DTREE_CAT_DIR(idx, subset) : 0;
        sum += d * w;
        sum_abs += (d & 1) * w;
        dir[i] = (char)d;
      }

      R = (sum_abs + sum) * 0.5;
      L = (sum_abs - sum) * 0.5;
    }
  } else  // split on ordered var
  {
    int split_point = node->split->ord.split_point;
    int n1 = node->get_num_valid(vi);
    cv::AutoBuffer<uchar> inn_buf(
        n * (sizeof(int) * (data->have_priors ? 3 : 2) + sizeof(float)));
    auto* val_buf = (float*)inn_buf.data();
    int* sorted_buf = (int*)(val_buf + n);
    int* sample_idx_buf = sorted_buf + n;
    const float* val = nullptr;
    const int* sorted = nullptr;
    data->get_ord_var_data(node, vi, val_buf, sorted_buf, &val, &sorted,
                           sample_idx_buf);

    assert(0 <= split_point && split_point < n1 - 1);

    if (!data->have_priors) {
      for (i = 0; i <= split_point; i++) dir[sorted[i]] = (char)-1;
      for (; i < n1; i++) dir[sorted[i]] = (char)1;
      for (; i < n; i++) dir[sorted[i]] = (char)0;

      L = split_point - 1;
      R = n1 - split_point + 1;
    } else {
      const double* priors = data->priors_mult->data.db;
      int* responses_buf = sample_idx_buf + n;
      const int* responses = data->get_class_labels(node, responses_buf);
      L = R = 0;

      for (i = 0; i <= split_point; i++) {
        int idx = sorted[i];
        double w = priors[responses[idx]];
        dir[idx] = (char)-1;
        L += w;
      }

      for (; i < n1; i++) {
        int idx = sorted[i];
        double w = priors[responses[idx]];
        dir[idx] = (char)1;
        R += w;
      }

      for (; i < n; i++) dir[sorted[i]] = (char)0;
    }
  }
  node->maxlr = MAX(L, R);
  return node->split->quality / (L + R);
}

void deleter(CvDTreeSplit* ptr) { cv::fastFree(ptr); }

CvDTree::DTreeBestSplitFinder::DTreeBestSplitFinder(CvDTree* _tree,
                                                    CvDTreeNode* _node) {
  tree = _tree;
  node = _node;
  splitSize = tree->get_data()->split_heap->elem_size;

  bestSplit.reset((CvDTreeSplit*)cv::fastMalloc(splitSize), deleter);
  memset(bestSplit.get(), 0, splitSize);
  bestSplit->quality = -1;
  bestSplit->condensed_idx = INT_MIN;
  split.reset((CvDTreeSplit*)cv::fastMalloc(splitSize), deleter);
  memset(split.get(), 0, splitSize);
  // haveSplit = false;
}

void CvDTree::DTreeBestSplitFinder::operator()(const BlockedRange& range) {
  int vi = 0, vi1 = range.begin(), vi2 = range.end();
  int n = node->sample_count;
  CvDTreeTrainData* local_data = tree->get_data();
  cv::AutoBuffer<uchar> inn_buf(2 * n * (sizeof(int) + sizeof(float)));

  for (vi = vi1; vi < vi2; vi++) {
    CvDTreeSplit* res = nullptr;
    int ci = local_data->get_var_type(vi);
    if (node->get_num_valid(vi) <= 1) continue;

    if (local_data->is_classifier) {
      if (ci >= 0)
        res = tree->find_split_cat_class(node, vi, bestSplit->quality,
                                         split.get(), inn_buf.data());
      else
        res = tree->find_split_ord_class(node, vi, bestSplit->quality,
                                         split.get(), inn_buf.data());
    } else {
      if (ci >= 0)
        res = tree->find_split_cat_reg(node, vi, bestSplit->quality,
                                       split.get(), inn_buf.data());
      else
        res = tree->find_split_ord_reg(node, vi, bestSplit->quality,
                                       split.get(), inn_buf.data());
    }

    if (res && bestSplit->quality < split->quality)
      memcpy(bestSplit.get(), split.get(), splitSize);
  }
}

CvDTreeSplit* CvDTree::find_best_split(CvDTreeNode* node) {
  DTreeBestSplitFinder finder(this, node);

  parallel_reduce(BlockedRange(0, data->var_count), finder);

  CvDTreeSplit* bestSplit = nullptr;
  if (finder.bestSplit->quality > 0) {
    bestSplit = data->new_split_cat(0, -1.0f);
    memcpy(bestSplit, finder.bestSplit.get(), finder.splitSize);
  }

  return bestSplit;
}

CvDTreeSplit* CvDTree::find_split_ord_class(CvDTreeNode* node, int vi,
                                            float init_quality,
                                            CvDTreeSplit* _split,
                                            uchar* _ext_buf) {
  const float epsilon = FLT_EPSILON * 2;
  int n = node->sample_count;
  int n1 = node->get_num_valid(vi);
  int m = data->get_num_classes();

  int base_size = 2 * m * sizeof(int);
  cv::AutoBuffer<uchar> inn_buf(base_size);
  if (!_ext_buf)
    inn_buf.allocate(base_size + n * (3 * sizeof(int) + sizeof(float)));
  uchar* base_buf = inn_buf.data();
  uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;
  auto* values_buf = (float*)ext_buf;
  int* sorted_indices_buf = (int*)(values_buf + n);
  int* sample_indices_buf = sorted_indices_buf + n;
  const float* values = nullptr;
  const int* sorted_indices = nullptr;
  data->get_ord_var_data(node, vi, values_buf, sorted_indices_buf, &values,
                         &sorted_indices, sample_indices_buf);
  int* responses_buf = sample_indices_buf + n;
  const int* responses = data->get_class_labels(node, responses_buf);

  const int* rc0 = data->counts->data.i;
  int* lc = (int*)base_buf;
  int* rc = lc + m;
  int i = 0, best_i = -1;
  double lsum2 = 0, rsum2 = 0, best_val = init_quality;
  const double* priors = data->have_priors ? data->priors_mult->data.db : nullptr;

  // init arrays of class instance counters on both sides of the split
  for (i = 0; i < m; i++) {
    lc[i] = 0;
    rc[i] = rc0[i];
  }

  // compensate for missing values
  for (i = n1; i < n; i++) {
    rc[responses[sorted_indices[i]]]--;
  }

  if (!priors) {
    int L = 0, R = n1;

    for (i = 0; i < m; i++) rsum2 += (double)rc[i] * rc[i];

    for (i = 0; i < n1 - 1; i++) {
      int idx = responses[sorted_indices[i]];
      int lv = 0, rv = 0;
      L++;
      R--;
      lv = lc[idx];
      rv = rc[idx];
      lsum2 += lv * 2 + 1;
      rsum2 -= rv * 2 - 1;
      lc[idx] = lv + 1;
      rc[idx] = rv - 1;

      if (values[i] + epsilon < values[i + 1]) {
        double val = (lsum2 * R + rsum2 * L) / ((double)L * R);
        if (best_val < val) {
          best_val = val;
          best_i = i;
        }
      }
    }
  } else {
    double L = 0, R = 0;
    for (i = 0; i < m; i++) {
      double wv = rc[i] * priors[i];
      R += wv;
      rsum2 += wv * wv;
    }

    for (i = 0; i < n1 - 1; i++) {
      int idx = responses[sorted_indices[i]];
      int lv = 0, rv = 0;
      double p = priors[idx], p2 = p * p;
      L += p;
      R -= p;
      lv = lc[idx];
      rv = rc[idx];
      lsum2 += p2 * (lv * 2 + 1);
      rsum2 -= p2 * (rv * 2 - 1);
      lc[idx] = lv + 1;
      rc[idx] = rv - 1;

      if (values[i] + epsilon < values[i + 1]) {
        double val = (lsum2 * R + rsum2 * L) / ((double)L * R);
        if (best_val < val) {
          best_val = val;
          best_i = i;
        }
      }
    }
  }

  CvDTreeSplit* split = nullptr;
  if (best_i >= 0) {
    split = _split ? _split : data->new_split_ord(0, 0.0f, 0, 0, 0.0f);
    split->var_idx = vi;
    split->ord.c = (values[best_i] + values[best_i + 1]) * 0.5f;
    split->ord.split_point = best_i;
    split->inversed = 0;
    split->quality = (float)best_val;
  }
  return split;
}

void CvDTree::cluster_categories(const int* vectors, int n, int m, int* csums,
                                 int k, int* labels) {
  // TODO: consider adding priors (class weights) and sample weights to the
  // clustering algorithm
  int iters = 0, max_iters = 100;
  int i = 0, j = 0, idx = 0;
  cv::AutoBuffer<double> buf(n + k);
  double *v_weights = buf.data(), *c_weights = buf.data() + n;
  bool modified = true;
  cv::RNG* r = data->rng;

  // assign labels randomly
  for (i = 0; i < n; i++) {
    int sum = 0;
    const int* v = vectors + i * m;
    labels[i] = i < k ? i : r->uniform(0, k);

    // compute weight of each vector
    for (j = 0; j < m; j++) sum += v[j];
    v_weights[i] = sum ? 1. / sum : 0.;
  }

  for (i = 0; i < n; i++) {
    int i1 = (*r)(n);
    int i2 = (*r)(n);
    CV_SWAP(labels[i1], labels[i2], j);
  }

  for (iters = 0; iters <= max_iters; iters++) {
    // calculate csums
    for (i = 0; i < k; i++) {
      for (j = 0; j < m; j++) csums[i * m + j] = 0;
    }

    for (i = 0; i < n; i++) {
      const int* v = vectors + i * m;
      int* s = csums + labels[i] * m;
      for (j = 0; j < m; j++) s[j] += v[j];
    }

    // exit the loop here, when we have up-to-date csums
    if (iters == max_iters || !modified) break;

    modified = false;

    // calculate weight of each cluster
    for (i = 0; i < k; i++) {
      const int* s = csums + i * m;
      int sum = 0;
      for (j = 0; j < m; j++) sum += s[j];
      c_weights[i] = sum ? 1. / sum : 0;
    }

    // now for each vector determine the closest cluster
    for (i = 0; i < n; i++) {
      const int* v = vectors + i * m;
      double alpha = v_weights[i];
      double min_dist2 = DBL_MAX;
      int min_idx = -1;

      for (idx = 0; idx < k; idx++) {
        const int* s = csums + idx * m;
        double dist2 = 0., beta = c_weights[idx];
        for (j = 0; j < m; j++) {
          double t = v[j] * alpha - s[j] * beta;
          dist2 += t * t;
        }
        if (min_dist2 > dist2) {
          min_dist2 = dist2;
          min_idx = idx;
        }
      }

      if (min_idx != labels[i]) modified = true;
      labels[i] = min_idx;
    }
  }
}

CvDTreeSplit* CvDTree::find_split_cat_class(CvDTreeNode* node, int vi,
                                            float init_quality,
                                            CvDTreeSplit* _split,
                                            uchar* _ext_buf) {
  int ci = data->get_var_type(vi);
  int n = node->sample_count;
  int m = data->get_num_classes();
  int _mi = data->cat_count->data.i[ci], mi = _mi;

  int base_size = sizeof(int) * m * (3 + mi) + sizeof(double) * (mi + 1);
  if (m > 2 && mi > data->params.max_categories)
    base_size +=
        (m * std::min(data->params.max_categories, n) + mi) * sizeof(int);
  else
    base_size += mi * sizeof(int*);
  cv::AutoBuffer<uchar> inn_buf(base_size);
  if (!_ext_buf) inn_buf.allocate(base_size + 2 * n * sizeof(int));
  uchar* base_buf = inn_buf.data();
  uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;

  int* lc = (int*)base_buf;
  int* rc = lc + m;
  int *_cjk = rc + m * 2, *cjk = _cjk;
  auto* c_weights = (double*)cv::alignPtr(cjk + m * mi, sizeof(double));

  int* labels_buf = (int*)ext_buf;
  const int* labels = data->get_cat_var_data(node, vi, labels_buf);
  int* responses_buf = labels_buf + n;
  const int* responses = data->get_class_labels(node, responses_buf);

  int* cluster_labels = nullptr;
  int** int_ptr = nullptr;
  int i = 0, j = 0, k = 0, idx = 0;
  double L = 0, R = 0;
  double best_val = init_quality;
  int prevcode = 0, best_subset = -1, subset_i = 0, subset_n = 0, subtract = 0;
  const double* priors = data->priors_mult->data.db;

  // init array of counters:
  // c_{jk} - number of samples that have vi-th input variable = j and response
  // = k.
  for (j = -1; j < mi; j++)
    for (k = 0; k < m; k++) cjk[j * m + k] = 0;

  for (i = 0; i < n; i++) {
    j = (labels[i] == 65535 && data->is_buf_16u) ? -1 : labels[i];
    k = responses[i];
    cjk[j * m + k]++;
  }

  if (m > 2) {
    if (mi > data->params.max_categories) {
      mi = MIN(data->params.max_categories, n);
      cjk = (int*)(c_weights + _mi);
      cluster_labels = cjk + m * mi;
      cluster_categories(_cjk, _mi, m, cjk, mi, cluster_labels);
    }
    subset_i = 1;
    subset_n = 1 << mi;
  } else {
    assert(m == 2);
    int_ptr = (int**)(c_weights + _mi);
    for (j = 0; j < mi; j++) int_ptr[j] = cjk + j * 2 + 1;
    std::sort(int_ptr, int_ptr + mi, LessThanPtr<int>());
    subset_i = 0;
    subset_n = mi;
  }

  for (k = 0; k < m; k++) {
    int sum = 0;
    for (j = 0; j < mi; j++) sum += cjk[j * m + k];
    rc[k] = sum;
    lc[k] = 0;
  }

  for (j = 0; j < mi; j++) {
    double sum = 0;
    for (k = 0; k < m; k++) sum += cjk[j * m + k] * priors[k];
    c_weights[j] = sum;
    R += c_weights[j];
  }

  for (; subset_i < subset_n; subset_i++) {
    double weight = NAN;
    int* crow = nullptr;
    double lsum2 = 0, rsum2 = 0;

    if (m == 2)
      idx = (int)(int_ptr[subset_i] - cjk) / 2;
    else {
      int graycode = (subset_i >> 1) ^ subset_i;
      int diff = graycode ^ prevcode;

      // determine index of the changed bit.
      Cv32suf u;
      idx = diff >= (1 << 16) ? 16 : 0;
      u.f = (float)(((diff >> 16) | diff) & 65535);
      idx += (u.i >> 23) - 127;
      subtract = graycode < prevcode;
      prevcode = graycode;
    }

    crow = cjk + idx * m;
    weight = c_weights[idx];
    if (weight < FLT_EPSILON) continue;

    if (!subtract) {
      for (k = 0; k < m; k++) {
        int t = crow[k];
        int lval = lc[k] + t;
        int rval = rc[k] - t;
        double p = priors[k], p2 = p * p;
        lsum2 += p2 * lval * lval;
        rsum2 += p2 * rval * rval;
        lc[k] = lval;
        rc[k] = rval;
      }
      L += weight;
      R -= weight;
    } else {
      for (k = 0; k < m; k++) {
        int t = crow[k];
        int lval = lc[k] - t;
        int rval = rc[k] + t;
        double p = priors[k], p2 = p * p;
        lsum2 += p2 * lval * lval;
        rsum2 += p2 * rval * rval;
        lc[k] = lval;
        rc[k] = rval;
      }
      L -= weight;
      R += weight;
    }

    if (L > FLT_EPSILON && R > FLT_EPSILON) {
      double val = (lsum2 * R + rsum2 * L) / ((double)L * R);
      if (best_val < val) {
        best_val = val;
        best_subset = subset_i;
      }
    }
  }

  CvDTreeSplit* split = nullptr;
  if (best_subset >= 0) {
    split = _split ? _split : data->new_split_cat(0, -1.0f);
    split->var_idx = vi;
    split->quality = (float)best_val;
    memset(split->subset, 0, (data->max_c_count + 31) / 32 * sizeof(int));
    if (m == 2) {
      for (i = 0; i <= best_subset; i++) {
        idx = (int)(int_ptr[i] - cjk) >> 1;
        split->subset[idx >> 5] |= 1 << (idx & 31);
      }
    } else {
      for (i = 0; i < _mi; i++) {
        idx = cluster_labels ? cluster_labels[i] : i;
        if (best_subset & (1 << idx)) split->subset[i >> 5] |= 1 << (i & 31);
      }
    }
  }
  return split;
}

CvDTreeSplit* CvDTree::find_split_ord_reg(CvDTreeNode* node, int vi,
                                          float init_quality,
                                          CvDTreeSplit* _split,
                                          uchar* _ext_buf) {
  const float epsilon = FLT_EPSILON * 2;
  int n = node->sample_count;
  int n1 = node->get_num_valid(vi);

  cv::AutoBuffer<uchar> inn_buf;
  if (!_ext_buf) inn_buf.allocate(2 * n * (sizeof(int) + sizeof(float)));
  uchar* ext_buf = _ext_buf ? _ext_buf : inn_buf.data();
  auto* values_buf = (float*)ext_buf;
  int* sorted_indices_buf = (int*)(values_buf + n);
  int* sample_indices_buf = sorted_indices_buf + n;
  const float* values = nullptr;
  const int* sorted_indices = nullptr;
  data->get_ord_var_data(node, vi, values_buf, sorted_indices_buf, &values,
                         &sorted_indices, sample_indices_buf);
  auto* responses_buf = (float*)(sample_indices_buf + n);
  const float* responses =
      data->get_ord_responses(node, responses_buf, sample_indices_buf);

  int i = 0, best_i = -1;
  double best_val = init_quality, lsum = 0, rsum = node->value * n;
  int L = 0, R = n1;

  // compensate for missing values
  for (i = n1; i < n; i++) rsum -= responses[sorted_indices[i]];

  // find the optimal split
  for (i = 0; i < n1 - 1; i++) {
    float t = responses[sorted_indices[i]];
    L++;
    R--;
    lsum += t;
    rsum -= t;

    if (values[i] + epsilon < values[i + 1]) {
      double val = (lsum * lsum * R + rsum * rsum * L) / ((double)L * R);
      if (best_val < val) {
        best_val = val;
        best_i = i;
      }
    }
  }

  CvDTreeSplit* split = nullptr;
  if (best_i >= 0) {
    split = _split ? _split : data->new_split_ord(0, 0.0f, 0, 0, 0.0f);
    split->var_idx = vi;
    split->ord.c = (values[best_i] + values[best_i + 1]) * 0.5f;
    split->ord.split_point = best_i;
    split->inversed = 0;
    split->quality = (float)best_val;
  }
  return split;
}

CvDTreeSplit* CvDTree::find_split_cat_reg(CvDTreeNode* node, int vi,
                                          float init_quality,
                                          CvDTreeSplit* _split,
                                          uchar* _ext_buf) {
  int ci = data->get_var_type(vi);
  int n = node->sample_count;
  int mi = data->cat_count->data.i[ci];

  int base_size =
      (mi + 2) * sizeof(double) + (mi + 1) * (sizeof(int) + sizeof(double*));
  cv::AutoBuffer<uchar> inn_buf(base_size);
  if (!_ext_buf)
    inn_buf.allocate(base_size + n * (2 * sizeof(int) + sizeof(float)));
  uchar* base_buf = inn_buf.data();
  uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;
  int* labels_buf = (int*)ext_buf;
  const int* labels = data->get_cat_var_data(node, vi, labels_buf);
  auto* responses_buf = (float*)(labels_buf + n);
  int* sample_indices_buf = (int*)(responses_buf + n);
  const float* responses =
      data->get_ord_responses(node, responses_buf, sample_indices_buf);

  double* sum = (double*)cv::alignPtr(base_buf, sizeof(double)) + 1;
  int* counts = (int*)(sum + mi) + 1;
  auto** sum_ptr = (double**)(counts + mi);
  int i = 0, L = 0, R = 0;
  double best_val = init_quality, lsum = 0, rsum = 0;
  int best_subset = -1, subset_i = 0;

  for (i = -1; i < mi; i++) sum[i] = counts[i] = 0;

  // calculate sum response and weight of each category of the input var
  for (i = 0; i < n; i++) {
    int idx = ((labels[i] == 65535) && data->is_buf_16u) ? -1 : labels[i];
    double s = sum[idx] + responses[i];
    int nc = counts[idx] + 1;
    sum[idx] = s;
    counts[idx] = nc;
  }

  // calculate average response in each category
  for (i = 0; i < mi; i++) {
    R += counts[i];
    rsum += sum[i];
    sum[i] /= MAX(counts[i], 1);
    sum_ptr[i] = sum + i;
  }

  std::sort(sum_ptr, sum_ptr + mi, LessThanPtr<double>());

  // revert back to unnormalized sums
  // (there should be a very little loss of accuracy)
  for (i = 0; i < mi; i++) sum[i] *= counts[i];

  for (subset_i = 0; subset_i < mi - 1; subset_i++) {
    int idx = (int)(sum_ptr[subset_i] - sum);
    int ni = counts[idx];

    if (ni) {
      double s = sum[idx];
      lsum += s;
      L += ni;
      rsum -= s;
      R -= ni;

      if (L && R) {
        double val = (lsum * lsum * R + rsum * rsum * L) / ((double)L * R);
        if (best_val < val) {
          best_val = val;
          best_subset = subset_i;
        }
      }
    }
  }

  CvDTreeSplit* split = nullptr;
  if (best_subset >= 0) {
    split = _split ? _split : data->new_split_cat(0, -1.0f);
    split->var_idx = vi;
    split->quality = (float)best_val;
    memset(split->subset, 0, (data->max_c_count + 31) / 32 * sizeof(int));
    for (i = 0; i <= best_subset; i++) {
      int idx = (int)(sum_ptr[i] - sum);
      split->subset[idx >> 5] |= 1 << (idx & 31);
    }
  }
  return split;
}

CvDTreeSplit* CvDTree::find_surrogate_split_ord(CvDTreeNode* node, int vi,
                                                uchar* _ext_buf) {
  const float epsilon = FLT_EPSILON * 2;
  const char* dir = (char*)data->direction->data.ptr;
  int n = node->sample_count, n1 = node->get_num_valid(vi);
  cv::AutoBuffer<uchar> inn_buf;
  if (!_ext_buf)
    inn_buf.allocate(
        n * (sizeof(int) * (data->have_priors ? 3 : 2) + sizeof(float)));
  uchar* ext_buf = _ext_buf ? _ext_buf : inn_buf.data();
  auto* values_buf = (float*)ext_buf;
  int* sorted_indices_buf = (int*)(values_buf + n);
  int* sample_indices_buf = sorted_indices_buf + n;
  const float* values = nullptr;
  const int* sorted_indices = nullptr;
  data->get_ord_var_data(node, vi, values_buf, sorted_indices_buf, &values,
                         &sorted_indices, sample_indices_buf);
  // LL - number of samples that both the primary and the surrogate splits send
  // to the left LR - ... primary split sends to the left and the surrogate
  // split sends to the right RL - ... primary split sends to the right and the
  // surrogate split sends to the left RR - ... both send to the right
  int i = 0, best_i = -1, best_inversed = 0;
  double best_val = NAN;

  if (!data->have_priors) {
    int LL = 0, RL = 0, LR = 0, RR = 0;
    int worst_val = cvFloor(node->maxlr), _best_val = worst_val;
    int sum = 0, sum_abs = 0;

    for (i = 0; i < n1; i++) {
      int d = dir[sorted_indices[i]];
      sum += d;
      sum_abs += d & 1;
    }

    // sum_abs = R + L; sum = R - L
    RR = (sum_abs + sum) >> 1;
    LR = (sum_abs - sum) >> 1;

    // initially all the samples are sent to the right by the surrogate split,
    // LR of them are sent to the left by primary split, and RR - to the right.
    // now iteratively compute LL, LR, RL and RR for every possible surrogate
    // split value.
    for (i = 0; i < n1 - 1; i++) {
      int d = dir[sorted_indices[i]];

      if (d < 0) {
        LL++;
        LR--;
        if (LL + RR > _best_val && values[i] + epsilon < values[i + 1]) {
          best_val = LL + RR;
          best_i = i;
          best_inversed = 0;
        }
      } else if (d > 0) {
        RL++;
        RR--;
        if (RL + LR > _best_val && values[i] + epsilon < values[i + 1]) {
          best_val = RL + LR;
          best_i = i;
          best_inversed = 1;
        }
      }
    }
    best_val = _best_val;
  } else {
    double LL = 0, RL = 0, LR = NAN, RR = NAN;
    double worst_val = node->maxlr;
    double sum = 0, sum_abs = 0;
    const double* priors = data->priors_mult->data.db;
    int* responses_buf = sample_indices_buf + n;
    const int* responses = data->get_class_labels(node, responses_buf);
    best_val = worst_val;

    for (i = 0; i < n1; i++) {
      int idx = sorted_indices[i];
      double w = priors[responses[idx]];
      int d = dir[idx];
      sum += d * w;
      sum_abs += (d & 1) * w;
    }

    // sum_abs = R + L; sum = R - L
    RR = (sum_abs + sum) * 0.5;
    LR = (sum_abs - sum) * 0.5;

    // initially all the samples are sent to the right by the surrogate split,
    // LR of them are sent to the left by primary split, and RR - to the right.
    // now iteratively compute LL, LR, RL and RR for every possible surrogate
    // split value.
    for (i = 0; i < n1 - 1; i++) {
      int idx = sorted_indices[i];
      double w = priors[responses[idx]];
      int d = dir[idx];

      if (d < 0) {
        LL += w;
        LR -= w;
        if (LL + RR > best_val && values[i] + epsilon < values[i + 1]) {
          best_val = LL + RR;
          best_i = i;
          best_inversed = 0;
        }
      } else if (d > 0) {
        RL += w;
        RR -= w;
        if (RL + LR > best_val && values[i] + epsilon < values[i + 1]) {
          best_val = RL + LR;
          best_i = i;
          best_inversed = 1;
        }
      }
    }
  }
  return best_i >= 0 && best_val > node->maxlr
             ? data->new_split_ord(vi,
                                   (values[best_i] + values[best_i + 1]) * 0.5f,
                                   best_i, best_inversed, (float)best_val)
             : nullptr;
}

CvDTreeSplit* CvDTree::find_surrogate_split_cat(CvDTreeNode* node, int vi,
                                                uchar* _ext_buf) {
  const char* dir = (char*)data->direction->data.ptr;
  int n = node->sample_count;
  int i = 0, mi = data->cat_count->data.i[data->get_var_type(vi)], l_win = 0;

  int base_size = (2 * (mi + 1) + 1) * sizeof(double) +
                  (!data->have_priors ? 2 * (mi + 1) * sizeof(int) : 0);
  cv::AutoBuffer<uchar> inn_buf(base_size);
  if (!_ext_buf)
    inn_buf.allocate(base_size +
                     n * (sizeof(int) + (data->have_priors ? sizeof(int) : 0)));
  uchar* base_buf = inn_buf.data();
  uchar* ext_buf = _ext_buf ? _ext_buf : base_buf + base_size;

  int* labels_buf = (int*)ext_buf;
  const int* labels = data->get_cat_var_data(node, vi, labels_buf);
  // LL - number of samples that both the primary and the surrogate splits send
  // to the left LR - ... primary split sends to the left and the surrogate
  // split sends to the right RL - ... primary split sends to the right and the
  // surrogate split sends to the left RR - ... both send to the right
  CvDTreeSplit* split = data->new_split_cat(vi, 0);
  double best_val = 0;
  double* lc = (double*)cv::alignPtr(base_buf, sizeof(double)) + 1;
  double* rc = lc + mi + 1;

  for (i = -1; i < mi; i++) lc[i] = rc[i] = 0;

  // for each category calculate the weight of samples
  // sent to the left (lc) and to the right (rc) by the primary split
  if (!data->have_priors) {
    int* _lc = (int*)rc + 1;
    int* _rc = _lc + mi + 1;

    for (i = -1; i < mi; i++) _lc[i] = _rc[i] = 0;

    for (i = 0; i < n; i++) {
      int idx = ((labels[i] == 65535) && (data->is_buf_16u)) ? -1 : labels[i];
      int d = dir[i];
      int sum = _lc[idx] + d;
      int sum_abs = _rc[idx] + (d & 1);
      _lc[idx] = sum;
      _rc[idx] = sum_abs;
    }

    for (i = 0; i < mi; i++) {
      int sum = _lc[i];
      int sum_abs = _rc[i];
      lc[i] = (sum_abs - sum) >> 1;
      rc[i] = (sum_abs + sum) >> 1;
    }
  } else {
    const double* priors = data->priors_mult->data.db;
    int* responses_buf = labels_buf + n;
    const int* responses = data->get_class_labels(node, responses_buf);

    for (i = 0; i < n; i++) {
      int idx = ((labels[i] == 65535) && (data->is_buf_16u)) ? -1 : labels[i];
      double w = priors[responses[i]];
      int d = dir[i];
      double sum = lc[idx] + d * w;
      double sum_abs = rc[idx] + (d & 1) * w;
      lc[idx] = sum;
      rc[idx] = sum_abs;
    }

    for (i = 0; i < mi; i++) {
      double sum = lc[i];
      double sum_abs = rc[i];
      lc[i] = (sum_abs - sum) * 0.5;
      rc[i] = (sum_abs + sum) * 0.5;
    }
  }

  // 2. now form the split.
  // in each category send all the samples to the same direction as majority
  for (i = 0; i < mi; i++) {
    double lval = lc[i], rval = rc[i];
    if (lval > rval) {
      split->subset[i >> 5] |= 1 << (i & 31);
      best_val += lval;
      l_win++;
    } else
      best_val += rval;
  }

  split->quality = (float)best_val;
  if (split->quality <= node->maxlr || l_win == 0 || l_win == mi)
    cvSetRemoveByPtr(data->split_heap, split), split = nullptr;

  return split;
}

void CvDTree::calc_node_value(CvDTreeNode* node) {
  int i = 0, j = 0, k = 0, n = node->sample_count, cv_n = data->params.cv_folds;
  int m = data->get_num_classes();

  int base_size = data->is_classifier
                      ? sizeof(int) * m * cv_n
                      : sizeof(double) * 2 * cv_n + sizeof(int) * cv_n;
  int ext_size =
      n * (sizeof(int) +
           (data->is_classifier ? sizeof(int) : sizeof(int) + sizeof(float)));
  cv::AutoBuffer<uchar> inn_buf(base_size + ext_size);
  uchar* base_buf = inn_buf.data();
  uchar* ext_buf = base_buf + base_size;

  int* cv_labels_buf = (int*)ext_buf;
  const int* cv_labels = data->get_cv_labels(node, cv_labels_buf);

  if (data->is_classifier) {
    // in case of classification tree:
    //  * node value is the label of the class that has the largest weight in
    //  the node.
    //  * node risk is the weighted number of misclassified samples,
    //  * j-th cross-validation fold value and risk are calculated as above,
    //    but using the samples with cv_labels(*)!=j.
    //  * j-th cross-validation fold error is calculated as the weighted number
    //  of
    //    misclassified samples with cv_labels(*)==j.

    // compute the number of instances of each class
    int* cls_count = data->counts->data.i;
    int* responses_buf = cv_labels_buf + n;
    const int* responses = data->get_class_labels(node, responses_buf);
    int* cv_cls_count = (int*)base_buf;
    double max_val = -1, total_weight = 0;
    int max_k = -1;
    double* priors = data->priors_mult->data.db;

    for (k = 0; k < m; k++) cls_count[k] = 0;

    if (cv_n == 0) {
      for (i = 0; i < n; i++) cls_count[responses[i]]++;
    } else {
      for (j = 0; j < cv_n; j++)
        for (k = 0; k < m; k++) cv_cls_count[j * m + k] = 0;

      for (i = 0; i < n; i++) {
        j = cv_labels[i];
        k = responses[i];
        cv_cls_count[j * m + k]++;
      }

      for (j = 0; j < cv_n; j++)
        for (k = 0; k < m; k++) cls_count[k] += cv_cls_count[j * m + k];
    }

    if (data->have_priors && node->parent == nullptr) {
      // compute priors_mult from priors, take the sample ratio into account.
      double sum = 0;
      for (k = 0; k < m; k++) {
        int n_k = cls_count[k];
        priors[k] = data->priors->data.db[k] * (n_k ? 1. / n_k : 0.);
        sum += priors[k];
      }
      sum = 1. / sum;
      for (k = 0; k < m; k++) priors[k] *= sum;
    }

    for (k = 0; k < m; k++) {
      double val = cls_count[k] * priors[k];
      total_weight += val;
      if (max_val < val) {
        max_val = val;
        max_k = k;
      }
    }

    node->class_idx = max_k;
    node->value = data->cat_map->data
                      .i[data->cat_ofs->data.i[data->cat_var_count] + max_k];
    node->node_risk = total_weight - max_val;

    for (j = 0; j < cv_n; j++) {
      double sum_k = 0, sum = 0, max_val_k = 0;
      max_val = -1;
      max_k = -1;

      for (k = 0; k < m; k++) {
        double w = priors[k];
        double val_k = cv_cls_count[j * m + k] * w;
        double val = cls_count[k] * w - val_k;
        sum_k += val_k;
        sum += val;
        if (max_val < val) {
          max_val = val;
          max_val_k = val_k;
          max_k = k;
        }
      }

      node->cv_Tn[j] = INT_MAX;
      node->cv_node_risk[j] = sum - max_val;
      node->cv_node_error[j] = sum_k - max_val_k;
    }
  } else {
    // in case of regression tree:
    //  * node value is 1/n*sum_i(Y_i), where Y_i is i-th response,
    //    n is the number of samples in the node.
    //  * node risk is the sum of squared errors: sum_i((Y_i - <node_value>)^2)
    //  * j-th cross-validation fold value and risk are calculated as above,
    //    but using the samples with cv_labels(*)!=j.
    //  * j-th cross-validation fold error is calculated
    //    using samples with cv_labels(*)==j as the test subset:
    //    error_j = sum_(i,cv_labels(i)==j)((Y_i - <node_value_j>)^2),
    //    where node_value_j is the node value calculated
    //    as described in the previous bullet, and summation is done
    //    over the samples with cv_labels(*)==j.

    double sum = 0, sum2 = 0;
    auto* values_buf = (float*)(cv_labels_buf + n);
    int* sample_indices_buf = (int*)(values_buf + n);
    const float* values =
        data->get_ord_responses(node, values_buf, sample_indices_buf);
    double *cv_sum = nullptr, *cv_sum2 = nullptr;
    int* cv_count = nullptr;

    if (cv_n == 0) {
      for (i = 0; i < n; i++) {
        double t = values[i];
        sum += t;
        sum2 += t * t;
      }
    } else {
      cv_sum = (double*)base_buf;
      cv_sum2 = cv_sum + cv_n;
      cv_count = (int*)(cv_sum2 + cv_n);

      for (j = 0; j < cv_n; j++) {
        cv_sum[j] = cv_sum2[j] = 0.;
        cv_count[j] = 0;
      }

      for (i = 0; i < n; i++) {
        j = cv_labels[i];
        double t = values[i];
        double s = cv_sum[j] + t;
        double s2 = cv_sum2[j] + t * t;
        int nc = cv_count[j] + 1;
        cv_sum[j] = s;
        cv_sum2[j] = s2;
        cv_count[j] = nc;
      }

      for (j = 0; j < cv_n; j++) {
        sum += cv_sum[j];
        sum2 += cv_sum2[j];
      }
    }

    node->node_risk = sum2 - (sum / n) * sum;
    node->value = sum / n;

    for (j = 0; j < cv_n; j++) {
      double s = cv_sum[j], si = sum - s;
      double s2 = cv_sum2[j], s2i = sum2 - s2;
      int c = cv_count[j], ci = n - c;
      double r = si / MAX(ci, 1);
      node->cv_node_risk[j] = s2i - r * r * ci;
      node->cv_node_error[j] = s2 - 2 * r * s + c * r * r;
      node->cv_Tn[j] = INT_MAX;
    }
  }
}

void CvDTree::complete_node_dir(CvDTreeNode* node) {
  int vi = 0, i = 0, n = node->sample_count, nl = 0, nr = 0, d0 = 0, d1 = -1;
  int nz = n - node->get_num_valid(node->split->var_idx);
  char* dir = (char*)data->direction->data.ptr;

  // try to complete direction using surrogate splits
  if (nz && data->params.use_surrogates) {
    cv::AutoBuffer<uchar> inn_buf(n * (2 * sizeof(int) + sizeof(float)));
    CvDTreeSplit* split = node->split->next;
    for (; split != nullptr && nz; split = split->next) {
      int inversed_mask = split->inversed ? -1 : 0;
      vi = split->var_idx;

      if (data->get_var_type(vi) >= 0)  // split on categorical var
      {
        int* labels_buf = (int*)inn_buf.data();
        const int* labels = data->get_cat_var_data(node, vi, labels_buf);
        const int* subset = split->subset;

        for (i = 0; i < n; i++) {
          int idx = labels[i];
          if (!dir[i] && (((idx >= 0) && (!data->is_buf_16u)) ||
                          ((idx != 65535) && (data->is_buf_16u))))

          {
            int d = CV_DTREE_CAT_DIR(idx, subset);
            dir[i] = (char)((d ^ inversed_mask) - inversed_mask);
            if (--nz) break;
          }
        }
      } else  // split on ordered var
      {
        auto* values_buf = (float*)inn_buf.data();
        int* sorted_indices_buf = (int*)(values_buf + n);
        int* sample_indices_buf = sorted_indices_buf + n;
        const float* values = nullptr;
        const int* sorted_indices = nullptr;
        data->get_ord_var_data(node, vi, values_buf, sorted_indices_buf,
                               &values, &sorted_indices, sample_indices_buf);
        int split_point = split->ord.split_point;
        int n1 = node->get_num_valid(vi);

        assert(0 <= split_point && split_point < n - 1);

        for (i = 0; i < n1; i++) {
          int idx = sorted_indices[i];
          if (!dir[idx]) {
            int d = i <= split_point ? -1 : 1;
            dir[idx] = (char)((d ^ inversed_mask) - inversed_mask);
            if (--nz) break;
          }
        }
      }
    }
  }

  // find the default direction for the rest
  if (nz) {
    for (i = nr = 0; i < n; i++) nr += dir[i] > 0;
    nl = n - nr - nz;
    d0 = nl > nr ? -1 : nr > nl;
  }

  // make sure that every sample is directed either to the left or to the right
  for (i = 0; i < n; i++) {
    int d = dir[i];
    if (!d) {
      d = d0;
      if (!d) d = d1, d1 = -d1;
    }
    d = d > 0;
    dir[i] = (char)d;  // remap (-1,1) to (0,1)
  }
}

void CvDTree::split_node_data(CvDTreeNode* node) {
  int vi = 0, i = 0, n = node->sample_count, nl = 0, nr = 0, scount = data->sample_count;
  char* dir = (char*)data->direction->data.ptr;
  CvDTreeNode *left = nullptr, *right = nullptr;
  int* new_idx = data->split_buf->data.i;
  int new_buf_idx = data->get_child_buf_idx(node);
  int work_var_count = data->get_work_var_count();
  CvMat* buf = data->buf;
  size_t length_buf_row = data->get_length_subbuf();
  cv::AutoBuffer<uchar> inn_buf(n * (3 * sizeof(int) + sizeof(float)));
  int* temp_buf = (int*)inn_buf.data();

  complete_node_dir(node);

  for (i = nl = nr = 0; i < n; i++) {
    int d = dir[i];
    // initialize new indices for splitting ordered variables
    new_idx[i] = (nl & (d - 1)) | (nr & -d);  // d ? ri : li
    nr += d;
    nl += d ^ 1;
  }

  bool split_input_data = false;
  node->left = left = data->new_node(node, nl, new_buf_idx, node->offset);
  node->right = right =
      data->new_node(node, nr, new_buf_idx, node->offset + nl);

  split_input_data =
      node->depth + 1 < data->params.max_depth &&
      (node->left->sample_count > data->params.min_sample_count ||
       node->right->sample_count > data->params.min_sample_count);

  // split ordered variables, keep both halves sorted.
  for (vi = 0; vi < data->var_count; vi++) {
    int ci = data->get_var_type(vi);

    if (ci >= 0 || !split_input_data) continue;

    int n1 = node->get_num_valid(vi);
    auto* src_val_buf = (float*)(uchar*)(temp_buf + n);
    int* src_sorted_idx_buf = (int*)(src_val_buf + n);
    int* src_sample_idx_buf = src_sorted_idx_buf + n;
    const float* src_val = nullptr;
    const int* src_sorted_idx = nullptr;
    data->get_ord_var_data(node, vi, src_val_buf, src_sorted_idx_buf, &src_val,
                           &src_sorted_idx, src_sample_idx_buf);

    for (i = 0; i < n; i++) temp_buf[i] = src_sorted_idx[i];

    if (data->is_buf_16u) {
      unsigned short *ldst = nullptr, *rdst = nullptr, *ldst0 = nullptr, *rdst0 = nullptr;
      // unsigned short tl, tr;
      ldst0 = ldst =
          (unsigned short*)(buf->data.s + left->buf_idx * length_buf_row +
                            vi * scount + left->offset);
      rdst0 = rdst = (unsigned short*)(ldst + nl);

      // split sorted
      for (i = 0; i < n1; i++) {
        int idx = temp_buf[i];
        int d = dir[idx];
        idx = new_idx[idx];
        if (d) {
          *rdst = (unsigned short)idx;
          rdst++;
        } else {
          *ldst = (unsigned short)idx;
          ldst++;
        }
      }

      left->set_num_valid(vi, (int)(ldst - ldst0));
      right->set_num_valid(vi, (int)(rdst - rdst0));

      // split missing
      for (; i < n; i++) {
        int idx = temp_buf[i];
        int d = dir[idx];
        idx = new_idx[idx];
        if (d) {
          *rdst = (unsigned short)idx;
          rdst++;
        } else {
          *ldst = (unsigned short)idx;
          ldst++;
        }
      }
    } else {
      int *ldst0 = nullptr, *ldst = nullptr, *rdst0 = nullptr, *rdst = nullptr;
      ldst0 = ldst = buf->data.i + left->buf_idx * length_buf_row +
                     vi * scount + left->offset;
      rdst0 = rdst = buf->data.i + right->buf_idx * length_buf_row +
                     vi * scount + right->offset;

      // split sorted
      for (i = 0; i < n1; i++) {
        int idx = temp_buf[i];
        int d = dir[idx];
        idx = new_idx[idx];
        if (d) {
          *rdst = idx;
          rdst++;
        } else {
          *ldst = idx;
          ldst++;
        }
      }

      left->set_num_valid(vi, (int)(ldst - ldst0));
      right->set_num_valid(vi, (int)(rdst - rdst0));

      // split missing
      for (; i < n; i++) {
        int idx = temp_buf[i];
        int d = dir[idx];
        idx = new_idx[idx];
        if (d) {
          *rdst = idx;
          rdst++;
        } else {
          *ldst = idx;
          ldst++;
        }
      }
    }
  }

  // split categorical vars, responses and cv_labels using new_idx relocation
  // table
  for (vi = 0; vi < work_var_count; vi++) {
    int ci = data->get_var_type(vi);
    int n1 = node->get_num_valid(vi), nr1 = 0;

    if (ci < 0 || (vi < data->var_count && !split_input_data)) continue;

    int* src_lbls_buf = temp_buf + n;
    const int* src_lbls = data->get_cat_var_data(node, vi, src_lbls_buf);

    for (i = 0; i < n; i++) temp_buf[i] = src_lbls[i];

    if (data->is_buf_16u) {
      auto* ldst =
          (unsigned short*)(buf->data.s + left->buf_idx * length_buf_row +
                            vi * scount + left->offset);
      auto* rdst =
          (unsigned short*)(buf->data.s + right->buf_idx * length_buf_row +
                            vi * scount + right->offset);

      for (i = 0; i < n; i++) {
        int d = dir[i];
        int idx = temp_buf[i];
        if (d) {
          *rdst = (unsigned short)idx;
          rdst++;
          nr1 += (idx != 65535) & d;
        } else {
          *ldst = (unsigned short)idx;
          ldst++;
        }
      }

      if (vi < data->var_count) {
        left->set_num_valid(vi, n1 - nr1);
        right->set_num_valid(vi, nr1);
      }
    } else {
      int* ldst = buf->data.i + left->buf_idx * length_buf_row + vi * scount +
                  left->offset;
      int* rdst = buf->data.i + right->buf_idx * length_buf_row + vi * scount +
                  right->offset;

      for (i = 0; i < n; i++) {
        int d = dir[i];
        int idx = temp_buf[i];
        if (d) {
          *rdst = idx;
          rdst++;
          nr1 += (idx >= 0) & d;
        } else {
          *ldst = idx;
          ldst++;
        }
      }

      if (vi < data->var_count) {
        left->set_num_valid(vi, n1 - nr1);
        right->set_num_valid(vi, nr1);
      }
    }
  }

  // split sample indices
  int* sample_idx_src_buf = temp_buf + n;
  const int* sample_idx_src =
      data->get_sample_indices(node, sample_idx_src_buf);

  for (i = 0; i < n; i++) temp_buf[i] = sample_idx_src[i];

  int pos = data->get_work_var_count();
  if (data->is_buf_16u) {
    auto* ldst =
        (unsigned short*)(buf->data.s + left->buf_idx * length_buf_row +
                          pos * scount + left->offset);
    auto* rdst =
        (unsigned short*)(buf->data.s + right->buf_idx * length_buf_row +
                          pos * scount + right->offset);
    for (i = 0; i < n; i++) {
      int d = dir[i];
      auto idx = (unsigned short)temp_buf[i];
      if (d) {
        *rdst = idx;
        rdst++;
      } else {
        *ldst = idx;
        ldst++;
      }
    }
  } else {
    int* ldst = buf->data.i + left->buf_idx * length_buf_row + pos * scount +
                left->offset;
    int* rdst = buf->data.i + right->buf_idx * length_buf_row + pos * scount +
                right->offset;
    for (i = 0; i < n; i++) {
      int d = dir[i];
      int idx = temp_buf[i];
      if (d) {
        *rdst = idx;
        rdst++;
      } else {
        *ldst = idx;
        ldst++;
      }
    }
  }

  // deallocate the parent node data that is not needed anymore
  data->free_node_data(node);
}

void CvDTree::prune_cv() {
  CvMat* ab = nullptr;
  CvMat* temp = nullptr;
  CvMat* err_jk = nullptr;

  // 1. build tree sequence for each cv fold, calculate error_{Tj,beta_k}.
  // 2. choose the best tree index (if need, apply 1SE rule).
  // 3. store the best index and cut the branches.

  CV_FUNCNAME("CvDTree::prune_cv");

  __CV_BEGIN__;

  int ti = 0, j = 0, tree_count = 0, cv_n = data->params.cv_folds,
             n = root->sample_count;
  // currently, 1SE for regression is not implemented
  bool use_1se = data->params.use_1se_rule != 0 && data->is_classifier;
  double* err = nullptr;
  double min_err = 0, min_err_se = 0;
  int min_idx = -1;

  CV_CALL(ab = cvCreateMat(1, 256, CV_64F));

  // build the main tree sequence, calculate alpha's
  for (;; tree_count++) {
    double min_alpha = update_tree_rnc(tree_count, -1);
    if (cut_tree(tree_count, -1, min_alpha)) break;

    if (ab->cols <= tree_count) {
      CV_CALL(temp = cvCreateMat(1, ab->cols * 3 / 2, CV_64F));
      for (ti = 0; ti < ab->cols; ti++) temp->data.db[ti] = ab->data.db[ti];
      cvReleaseMat(&ab);
      ab = temp;
      temp = nullptr;
    }

    ab->data.db[tree_count] = min_alpha;
  }

  ab->data.db[0] = 0.;

  if (tree_count > 0) {
    for (ti = 1; ti < tree_count - 1; ti++)
      ab->data.db[ti] = sqrt(ab->data.db[ti] * ab->data.db[ti + 1]);
    ab->data.db[tree_count - 1] = DBL_MAX * 0.5;

    CV_CALL(err_jk = cvCreateMat(cv_n, tree_count, CV_64F));
    err = err_jk->data.db;

    for (j = 0; j < cv_n; j++) {
      int tj = 0, tk = 0;
      for (; tk < tree_count; tj++) {
        double min_alpha = update_tree_rnc(tj, j);
        if (cut_tree(tj, j, min_alpha)) min_alpha = DBL_MAX;

        for (; tk < tree_count; tk++) {
          if (ab->data.db[tk] > min_alpha) break;
          err[j * tree_count + tk] = root->tree_error;
        }
      }
    }

    for (ti = 0; ti < tree_count; ti++) {
      double sum_err = 0;
      for (j = 0; j < cv_n; j++) sum_err += err[j * tree_count + ti];
      if (ti == 0 || sum_err < min_err) {
        min_err = sum_err;
        min_idx = ti;
        if (use_1se) min_err_se = sqrt(sum_err * (n - sum_err));
      } else if (sum_err < min_err + min_err_se)
        min_idx = ti;
    }
  }

  pruned_tree_idx = min_idx;
  free_prune_data(data->params.truncate_pruned_tree != 0);

  __CV_END__;

  cvReleaseMat(&err_jk);
  cvReleaseMat(&ab);
  cvReleaseMat(&temp);
}

double CvDTree::update_tree_rnc(int T, int fold) {
  CvDTreeNode* node = root;
  double min_alpha = DBL_MAX;

  for (;;) {
    CvDTreeNode* parent = nullptr;
    for (;;) {
      int t = fold >= 0 ? node->cv_Tn[fold] : node->Tn;
      if (t <= T || !node->left) {
        node->complexity = 1;
        node->tree_risk = node->node_risk;
        node->tree_error = 0.;
        if (fold >= 0) {
          node->tree_risk = node->cv_node_risk[fold];
          node->tree_error = node->cv_node_error[fold];
        }
        break;
      }
      node = node->left;
    }

    for (parent = node->parent; parent && parent->right == node;
         node = parent, parent = parent->parent) {
      parent->complexity += node->complexity;
      parent->tree_risk += node->tree_risk;
      parent->tree_error += node->tree_error;

      parent->alpha =
          ((fold >= 0 ? parent->cv_node_risk[fold] : parent->node_risk) -
           parent->tree_risk) /
          (parent->complexity - 1);
      min_alpha = MIN(min_alpha, parent->alpha);
    }

    if (!parent) break;

    parent->complexity = node->complexity;
    parent->tree_risk = node->tree_risk;
    parent->tree_error = node->tree_error;
    node = parent->right;
  }

  return min_alpha;
}

int CvDTree::cut_tree(int T, int fold, double min_alpha) {
  CvDTreeNode* node = root;
  if (!node->left) return 1;

  for (;;) {
    CvDTreeNode* parent = nullptr;
    for (;;) {
      int t = fold >= 0 ? node->cv_Tn[fold] : node->Tn;
      if (t <= T || !node->left) break;
      if (node->alpha <= min_alpha + FLT_EPSILON) {
        if (fold >= 0)
          node->cv_Tn[fold] = T;
        else
          node->Tn = T;
        if (node == root) return 1;
        break;
      }
      node = node->left;
    }

    for (parent = node->parent; parent && parent->right == node;
         node = parent, parent = parent->parent)
      ;

    if (!parent) break;

    node = parent->right;
  }

  return 0;
}

void CvDTree::free_prune_data(bool _cut_tree) {
  CvDTreeNode* node = root;

  for (;;) {
    CvDTreeNode* parent = nullptr;
    for (;;) {
      // do not call cvSetRemoveByPtr( cv_heap, node->cv_Tn )
      // as we will clear the whole cross-validation heap at the end
      node->cv_Tn = nullptr;
      node->cv_node_error = node->cv_node_risk = nullptr;
      if (!node->left) break;
      node = node->left;
    }

    for (parent = node->parent; parent && parent->right == node;
         node = parent, parent = parent->parent) {
      if (_cut_tree && parent->Tn <= pruned_tree_idx) {
        data->free_node(parent->left);
        data->free_node(parent->right);
        parent->left = parent->right = nullptr;
      }
    }

    if (!parent) break;

    node = parent->right;
  }

  if (data->cv_heap) cvClearSet(data->cv_heap);
}

void CvDTree::free_tree() {
  if (root && data && data->shared) {
    pruned_tree_idx = INT_MIN;
    free_prune_data(true);
    data->free_node(root);
    root = nullptr;
  }
}

CvDTreeNode* CvDTree::predict(const CvMat* _sample, const CvMat* _missing,
                              bool preprocessed_input) const {
  cv::AutoBuffer<int> catbuf;

  int i = 0, mstep = 0;
  const uchar* m = nullptr;
  CvDTreeNode* node = root;

  if (!node) CV_Error(CV_StsError, "The tree has not been trained yet");

  if (!CV_IS_MAT(_sample) || CV_MAT_TYPE(_sample->type) != CV_32FC1 ||
      (_sample->cols != 1 && _sample->rows != 1) ||
      (_sample->cols + _sample->rows - 1 != data->var_all &&
       !preprocessed_input) ||
      (_sample->cols + _sample->rows - 1 != data->var_count &&
       preprocessed_input))
    CV_Error(CV_StsBadArg,
             "the input sample must be 1d floating-point vector with the same "
             "number of elements as the total number of variables used for "
             "training");

  const float* sample = _sample->data.fl;
  int step =
      CV_IS_MAT_CONT(_sample->type) ? 1 : _sample->step / sizeof(sample[0]);

  if (data->cat_count &&
      !preprocessed_input)  // cache for categorical variables
  {
    int n = data->cat_count->cols;
    catbuf.allocate(n);
    for (i = 0; i < n; i++) catbuf[i] = -1;
  }

  if (_missing) {
    if (!CV_IS_MAT(_missing) || !CV_IS_MASK_ARR(_missing) ||
        !CV_ARE_SIZES_EQ(_missing, _sample))
      CV_Error(CV_StsBadArg,
               "the missing data mask must be 8-bit vector of the same size as "
               "input sample");
    m = _missing->data.ptr;
    mstep = CV_IS_MAT_CONT(_missing->type) ? 1 : _missing->step / sizeof(m[0]);
  }

  const int* vtype = data->var_type->data.i;
  const int* vidx =
      data->var_idx && !preprocessed_input ? data->var_idx->data.i : nullptr;
  const int* cmap = data->cat_map ? data->cat_map->data.i : nullptr;
  const int* cofs = data->cat_ofs ? data->cat_ofs->data.i : nullptr;

  while (node->Tn > pruned_tree_idx && node->left) {
    CvDTreeSplit* split = node->split;
    int dir = 0;
    for (; !dir && split != nullptr; split = split->next) {
      int vi = split->var_idx;
      int ci = vtype[vi];
      i = vidx ? vidx[vi] : vi;
      float val = sample[(size_t)i * step];
      if (m && m[(size_t)i * mstep]) continue;
      if (ci < 0)  // ordered
        dir = val <= split->ord.c ? -1 : 1;
      else  // categorical
      {
        int c = 0;
        if (preprocessed_input)
          c = cvRound(val);
        else {
          c = catbuf[ci];
          if (c < 0) {
            int a = c = cofs[ci];
            int b = (ci + 1 >= data->cat_ofs->cols) ? data->cat_map->cols
                                                    : cofs[ci + 1];

            int ival = cvRound(val);
            if (ival != val)
              CV_Error(CV_StsBadArg,
                       "one of input categorical variable is not an integer");

            while (a < b) {
              c = (a + b) >> 1;
              if (ival < cmap[c])
                b = c;
              else if (ival > cmap[c])
                a = c + 1;
              else
                break;
            }

            if (c < 0 || ival != cmap[c]) continue;

            catbuf[ci] = c -= cofs[ci];
          }
        }
        c = ((c == 65535) && data->is_buf_16u) ? -1 : c;
        dir = CV_DTREE_CAT_DIR(c, split->subset);
      }

      if (split->inversed) dir = -dir;
    }

    if (!dir) {
      double diff = node->right->sample_count - node->left->sample_count;
      dir = diff < 0 ? -1 : 1;
    }
    node = dir < 0 ? node->left : node->right;
  }

  return node;
}

CvDTreeNode* CvDTree::predict(const cv::Mat& _sample, const cv::Mat& _missing,
                              bool preprocessed_input) const {
  CvMat sample = cvMat(_sample), mmask = cvMat(_missing);
  return predict(&sample, mmask.data.ptr ? &mmask : nullptr, preprocessed_input);
}
