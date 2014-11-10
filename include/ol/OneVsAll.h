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
        int predict(FeatureVec feature_vec);
        void pushData(FeatureVec feature_vec,  int label);
    private:
        std::vector<Predictor> binary_predictors_;
        std::pair<int,int> binary_labels_;
    };
}