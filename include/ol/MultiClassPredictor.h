#pragma once
#include <ol/Constants.h>
#include <ol/Dataset.h>

namespace ol {

class MultiClassPredictor{
  public:
    virtual Label predict(const FeatureVec& feature_vec) = 0;
    virtual void pushData(const FeatureVec& feature_vec, Label label) = 0;
};

}
