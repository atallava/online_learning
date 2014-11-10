#pragma once

#include <vector>
#include <string>

#include <ol/Constants.h>
#include <ol/Predictor.h>
#include <ol/Logistic.h>
#include <ol/ExpGradDescent.h>

namespace ol {
    class OneVsAll {
        OneVsAll(int num_rounds, std::string type);
        int predict(const FeatureVec& feature_vec);
        void pushData(const FeatureVec& feature_vec, Label label);
    private:
        std::vector<Predictor> binary_predictors_;
        std::pair<int,int> binary_labels_;
    };
}