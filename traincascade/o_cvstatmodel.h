#pragma once

// CvCascadeBoost
class CvStatModel
{
public:
    CvStatModel();
    virtual ~CvStatModel();

    virtual void clear();
protected:
    const char* default_model_name;
};
