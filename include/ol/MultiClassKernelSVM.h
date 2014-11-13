#pragma once
#include <ol/MultiClassPredictor.h>
#include <set>

namespace ol {

class AlphaCompare{
  public:
    bool operator() (const std::pair<int,double>& lhs, const std::pair<int,double>& rhs) const{
      return fabs(lhs.second) < fabs(rhs.second);
    }
};

class MultiClassKernelSVM: public MultiClassPredictor{
  public:
    MultiClassKernelSVM(int num_rounds);
    Label predict(const FeatureVec& feature_vec);
    void pushData(const FeatureVec& feature_vec, Label label);
  private:
    double kernelFunction(int kernel_id, const FeatureVec& features);
    double RBF(const std::vector<double>& x1, const std::vector<double>& x2);

    std::vector<std::vector<double> > alpha_;
    std::vector<std::set<std::pair<int,double>, AlphaCompare> > best_alpha_;
    std::vector<FeatureVec> data_;
    double learning_rate_;
    int current_iteration_;
    double lambda_;
    double gamma_;
    int num_alpha_used_;
};

}
