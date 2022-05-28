#pragma once

#include "o_cvdtree.h"

class CvBoost;

// CvBoost, CvCascadeBoostTree
class CvBoostTree : public CvDTree {
 public:
  CvBoostTree();
  virtual ~CvBoostTree();

  virtual bool train(CvDTreeTrainData* trainData, const CvMat* subsample_idx,
                     CvBoost* ensemble);

  virtual void scale(double s);
  void clear() override;

  /* dummy methods to avoid warnings: BEGIN */
  bool train(const CvMat* trainData, int tflag, const CvMat* responses,
             const CvMat* varIdx = 0, const CvMat* sampleIdx = 0,
             const CvMat* varType = 0, const CvMat* missingDataMask = 0,
             CvDTreeParams params = CvDTreeParams()) override;
  bool train(CvDTreeTrainData* trainData, const CvMat* _subsample_idx) override;
  /* dummy methods to avoid warnings: END */

 protected:
  void try_split_node(CvDTreeNode* n) override;
  CvDTreeSplit* find_surrogate_split_ord(CvDTreeNode* n, int vi,
                                         uchar* ext_buf = 0) override;
  CvDTreeSplit* find_surrogate_split_cat(CvDTreeNode* n, int vi,
                                         uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_ord_class(CvDTreeNode* n, int vi,
                                     float init_quality = 0,
                                     CvDTreeSplit* _split = 0,
                                     uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_cat_class(CvDTreeNode* n, int vi,
                                     float init_quality = 0,
                                     CvDTreeSplit* _split = 0,
                                     uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_ord_reg(CvDTreeNode* n, int vi,
                                   float init_quality = 0,
                                   CvDTreeSplit* _split = 0,
                                   uchar* ext_buf = 0) override;
  CvDTreeSplit* find_split_cat_reg(CvDTreeNode* n, int vi,
                                   float init_quality = 0,
                                   CvDTreeSplit* _split = 0,
                                   uchar* ext_buf = 0) override;
  void calc_node_value(CvDTreeNode* n) override;
  double calc_node_dir(CvDTreeNode* n) override;

  CvBoost* ensemble;
};
