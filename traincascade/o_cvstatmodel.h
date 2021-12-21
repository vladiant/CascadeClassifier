#pragma once

// CvCascadeBoost
class CvStatModel
{
public:
    CvStatModel();
    virtual ~CvStatModel();

    virtual void clear();

    // CV_WRAP virtual void save( const char* filename, const char* name=0 ) const;
    // CV_WRAP virtual void load( const char* filename, const char* name=0 );

    // virtual void write( cv::FileStorage& storage, const char* name ) const;
    // virtual void read( const cv::FileNode& node );

protected:
    const char* default_model_name;
};
