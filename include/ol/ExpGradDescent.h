#pragma once
#include <vector>

#include <ol/Predictor.h>

namespace ol {
  class ExpGradDescent : public Predictor {
  public:
        ExpGradDescent(int num_rounds);
        int predict(const FeatureVec& feature_vec, double& confidence);
        void pushData(const FeatureVec& feature_vec,  int label);
        static std::pair<int,int> getBinaryLabels() { return std::pair<int,int> (-1,1) ; }
    private:
        void updateWeights();
        double threshold_;
        double learning_rate_;
        std::vector<double> weights_;
        double G_;
  };
}
