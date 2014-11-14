#pragma once

#include <vector>
#include <string>
#include <memory>

#include <ol/Constants.h>
#include <ol/Predictor.h>
#include <ol/Logistic.h>
#include <ol/ExpGradDescent.h>
#include <ol/MultiClassPredictor.h>

namespace ol {
    class OneVsAll : public MultiClassPredictor{
    public:
        OneVsAll(std::string type, MultiClassPredictorParams params);
        OneVsAll(int num_rounds, std::string type, double predictor_param);
        Label predict(const FeatureVec& feature_vec);
        void pushData(const FeatureVec& feature_vec, Label label);
    private:
        std::vector< std::shared_ptr<Predictor> > binary_predictors_;
        std::pair<int,int> binary_labels_;
    };
}
