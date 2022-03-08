#pragma once

class CvMat;

template<typename T, typename Idx>
class LessThanIdx
{
public:
    LessThanIdx( const T* _arr ) : arr(_arr) {}
    bool operator()(Idx a, Idx b) const { return arr[a] < arr[b]; }
    const T* arr;
};

template<typename T>
class LessThanPtr
{
public:
    bool operator()(T* a, T* b) const { return *a < *b; }
};

int icvCmpIntegers( const void* a, const void* b );

int cvAlign( int size, int align );

int CV_DTREE_CAT_DIR(int idx, const int* subset);

CvMat*
cvPreprocessIndexArray( const CvMat* idx_arr, int data_arr_size, bool check_for_duplicates=false );
