#pragma once
#include <vector>

#include <ol/Predictor.h>

namespace ol {
  class ExpGradDescent : public Predictor {
  public:
        ExpGradDescent(int num_features, int num_rounds);
        int predict(FeatureVec feature_vec);
        void pushData(FeatureVec feature_vec,  int label);
    private:
        void updateWeights();
        double threshold_;
        double learning_rate_;
        std::vector<double> weights_;
        double G_;
  };
}

