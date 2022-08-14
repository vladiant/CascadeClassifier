#pragma once

#include <cstddef>

#include "o_cvdtreeparams.h"

namespace cv {
class RNG;
}

struct CvMat;
struct CvDTreeNode;
struct CvDTreeSplit;
struct CvMemStorage;
struct CvSet;

// CvBoost
struct CvDTreeTrainData {
  CvDTreeTrainData();
  CvDTreeTrainData(const CvMat* trainData, int tflag, const CvMat* responses,
                   const CvMat* varIdx = 0, const CvMat* sampleIdx = 0,
                   const CvMat* varType = 0, const CvMat* missingDataMask = 0,
                   const CvDTreeParams& params = CvDTreeParams(),
                   bool _shared = false, bool _add_labels = false);
  virtual ~CvDTreeTrainData();

  virtual void set_data(const CvMat* trainData, int tflag,
                        const CvMat* responses, const CvMat* varIdx = 0,
                        const CvMat* sampleIdx = 0, const CvMat* varType = 0,
                        const CvMat* missingDataMask = 0,
                        const CvDTreeParams& params = CvDTreeParams(),
                        bool _shared = false, bool _add_labels = false,
                        bool _update_data = false);
  virtual void do_responses_copy();

  virtual void get_vectors(const CvMat* _subsample_idx, float* values,
                           unsigned char* missing, float* responses,
                           bool get_class_idx = false);

  virtual CvDTreeNode* subsample_data(const CvMat* _subsample_idx);

  // release all the data
  virtual void clear();

  int get_num_classes() const;
  int get_var_type(int vi) const;
  int get_work_var_count() const { return work_var_count; }

  virtual const float* get_ord_responses(CvDTreeNode* n, float* values_buf,
                                         int* sample_indices_buf);
  virtual const int* get_class_labels(CvDTreeNode* n, int* labels_buf);
  virtual const int* get_cv_labels(CvDTreeNode* n, int* labels_buf);
  virtual const int* get_sample_indices(CvDTreeNode* n, int* indices_buf);
  virtual const int* get_cat_var_data(CvDTreeNode* n, int vi,
                                      int* cat_values_buf);
  virtual void get_ord_var_data(CvDTreeNode* n, int vi, float* ord_values_buf,
                                int* sorted_indices_buf,
                                const float** ord_values,
                                const int** sorted_indices,
                                int* sample_indices_buf);
  virtual int get_child_buf_idx(CvDTreeNode* n);

  ////////////////////////////////////

  virtual bool set_params(const CvDTreeParams& params);
  virtual CvDTreeNode* new_node(CvDTreeNode* parent, int count, int storage_idx,
                                int offset);

  virtual CvDTreeSplit* new_split_ord(int vi, float cmp_val, int split_point,
                                      int inversed, float quality);
  virtual CvDTreeSplit* new_split_cat(int vi, float quality);
  virtual void free_node_data(CvDTreeNode* node);
  virtual void free_train_data();
  virtual void free_node(CvDTreeNode* node);

  int sample_count, var_all, var_count, max_c_count;
  int ord_var_count, cat_var_count, work_var_count;
  bool have_labels, have_priors;
  bool is_classifier;
  int tflag;

  const CvMat* train_data;
  const CvMat* responses;
  CvMat* responses_copy;  // used in Boosting

  int buf_count,
      buf_size;  // buf_size is obsolete, please do not use it, use expression
                 // ((int64)buf->rows * (int64)buf->cols / buf_count) instead
  bool shared;
  int is_buf_16u;

  CvMat* cat_count;
  CvMat* cat_ofs;
  CvMat* cat_map;

  CvMat* counts;
  CvMat* buf;
  inline size_t get_length_subbuf() const {
    size_t res = (size_t)(work_var_count + 1) * (size_t)sample_count;
    return res;
  }

  CvMat* direction;
  CvMat* split_buf;

  CvMat* var_idx;
  CvMat* var_type;  // i-th element =
                    //   k<0  - ordered
                    //   k>=0 - categorical, see k-th element of cat_* arrays
  CvMat* priors;
  CvMat* priors_mult;

  CvDTreeParams params;

  CvMemStorage* tree_storage;
  CvMemStorage* temp_storage;

  CvDTreeNode* data_root;

  CvSet* node_heap;
  CvSet* split_heap;
  CvSet* cv_heap;
  CvSet* nv_heap;

  cv::RNG* rng;
};