#pragma once
#include <ol/MultiClassPredictor.h>

namespace ol {

class MultiClassSVM: public MultiClassPredictor{
  public:
    MultiClassSVM(int num_rounds);
    Label predict(const FeatureVec& feature_vec);
    void pushData(const FeatureVec& feature_vec, Label label);
  private:
    std::vector<std::vector<double> > weights_;
    double learning_rate_;
    int current_iteration_;
    double lambda_;
};

}
