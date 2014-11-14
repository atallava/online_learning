#pragma once
#include <vector>

#include <ol/Dataset.h>
#include <ol/MultiClassPredictor.h>

namespace ol {
    class Predictor {
    public:
        virtual int predict(const FeatureVec& feature_vec, double& confidence) = 0;
        virtual void pushData(const FeatureVec& feature_vec,  int label) = 0;
        static std::pair<int,int> getBinaryLabels();
    };
}

