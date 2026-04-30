/**
 * @file o_cvdtreetraindata.h
 * @brief Shared training-data store backing @ref CvDTree and @ref CvBoost.
 *
 * @ref CvDTreeTrainData converts the public CvMat inputs (training data,
 * responses, optional sample / variable masks) into the internal
 * representation used by the split-finding algorithms: per-variable
 * sorted indices, categorical maps, sample-index buffers, and the
 * memory-pooled node / split / cv-data heaps. A single instance can be
 * shared across many @c CvDTree / @c CvBoostTree weak learners by
 * passing @c shared = true at construction.
 */
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

/**
 * @brief Internal data store for one or more decision trees.
 *
 * The class holds three categories of state:
 *  - immutable copies of the original training data (@c train_data,
 *    @c responses) and the metadata needed to interpret them
 *    (@c var_type, @c cat_count / @c cat_ofs / @c cat_map, @c priors);
 *  - working buffers reused across nodes (@c buf, @c counts,
 *    @c direction, @c split_buf);
 *  - memory pools for nodes, splits and pruning bookkeeping
 *    (@c tree_storage, @c temp_storage, @c node_heap, @c split_heap,
 *    @c cv_heap, @c nv_heap).
 */
struct CvDTreeTrainData {
  CvDTreeTrainData();
  /// Build the data store and immediately pre-process @p trainData.
  CvDTreeTrainData(const CvMat* trainData, int tflag, const CvMat* responses,
                   const CvMat* varIdx = 0, const CvMat* sampleIdx = 0,
                   const CvMat* varType = 0, const CvMat* missingDataMask = 0,
                   const CvDTreeParams& params = CvDTreeParams(),
                   bool _shared = false, bool _add_labels = false);
  virtual ~CvDTreeTrainData();

  /**
   * @brief Replace or initialize the underlying training data set.
   *
   * @param _shared When @c true the same instance can back multiple trees
   *        (used by boosting). Implies the data must outlive every tree
   *        that references it.
   * @param _add_labels Append a categorical label column synthesized from
   *        the responses; needed for some boosting variants.
   * @param _update_data When @c true reuse already-allocated buffers if
   *        the new data is layout-compatible; otherwise reallocate.
   */
  virtual void set_data(const CvMat* trainData, int tflag,
                        const CvMat* responses, const CvMat* varIdx = 0,
                        const CvMat* sampleIdx = 0, const CvMat* varType = 0,
                        const CvMat* missingDataMask = 0,
                        const CvDTreeParams& params = CvDTreeParams(),
                        bool _shared = false, bool _add_labels = false,
                        bool _update_data = false);
  /// Make a writable copy of the responses (boosting needs to mutate them).
  virtual void do_responses_copy();

  /// Materialize the dense (values, missing mask, responses) tuple for @p _subsample_idx.
  virtual void get_vectors(const CvMat* _subsample_idx, float* values,
                           unsigned char* missing, float* responses,
                           bool get_class_idx = false);

  /// Build the root node of a new tree from the supplied subsample.
  virtual CvDTreeNode* subsample_data(const CvMat* _subsample_idx);

  /// Release every internal buffer; safe to call on a default-constructed instance.
  virtual void clear();

  /// Number of distinct response classes (1 in regression mode).
  int get_num_classes() const;
  /// Type of variable @p vi: <0 = ordered, >=0 = categorical (index into cat_*).
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

  /// Validate parameters and propagate them to internal flags. Returns @c false on bad input.
  virtual bool set_params(const CvDTreeParams& params);
  /// Allocate and link a new tree node from the @c node_heap.
  virtual CvDTreeNode* new_node(CvDTreeNode* parent, int count, int storage_idx,
                                int offset);

  /// Allocate an ordered-variable split from the @c split_heap.
  virtual CvDTreeSplit* new_split_ord(int vi, float cmp_val, int split_point,
                                      int inversed, float quality);
  /// Allocate a categorical-variable split from the @c split_heap.
  virtual CvDTreeSplit* new_split_cat(int vi, float quality);
  /// Release per-node bookkeeping (pruning data, num_valid, ...).
  virtual void free_node_data(CvDTreeNode* node);
  /// Release the working buffers shared across nodes.
  virtual void free_train_data();
  /// Return one node to the @c node_heap (with its split).
  virtual void free_node(CvDTreeNode* node);

  int sample_count{}, var_all{}, var_count{}, max_c_count{};
  int ord_var_count{}, cat_var_count{}, work_var_count{};
  bool have_labels{}, have_priors{};
  bool is_classifier{}; ///< @c true when the response is categorical.
  int tflag{};

  const CvMat* train_data{};
  const CvMat* responses{};
  CvMat* responses_copy;  // used in Boosting

  int buf_count{},
      buf_size{};  // buf_size is obsolete, please do not use it, use expression
                 // ((int64)buf->rows * (int64)buf->cols / buf_count) instead
  bool shared{};
  int is_buf_16u{};

  CvMat* cat_count;
  CvMat* cat_ofs;
  CvMat* cat_map;

  CvMat* counts;
  CvMat* buf;
  /// Length (in elements) of one sub-buffer inside @c buf.
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

  CvMemStorage* tree_storage; ///< Pool for permanent tree nodes / splits.
  CvMemStorage* temp_storage; ///< Pool for scratch data freed at the end of training.

  CvDTreeNode* data_root{};

  CvSet* node_heap{};
  CvSet* split_heap{};
  CvSet* cv_heap{};
  CvSet* nv_heap{};

  cv::RNG* rng{};
};