#pragma once
#include <vector>

#include <ol/Dataset.h>

namespace ol {
    class Predictor {
    public:
        virtual int predict(FeatureVec feature_vec) = 0;
        virtual void pushData(FeatureVec feature_vec,  int label) = 0;
    };
}

