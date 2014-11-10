#pragma once
#include <vector>
#include <pf/Predictor.h>

namespace ol {
  class Winnow : public Predictor {
  public:
        Winnow(int num_features);
        int predict(FeatureVec feature_vec);
        void pushData(FeatureVec feature_vec,  int label);
    private:
        void updateWeights();
        double threshold_;
        std::vector<double> weights_;
  };
}

