#include <math.h>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ol/ExpGradDescent.h>

using namespace ol;

ExpGradDescent::ExpGradDescent(int num_features, int num_rounds) : weights_(num_features,1) 
{
    double G_ = 1;
    learning_rate_ = sqrt(std::log(num_features)/num_rounds)/G_;
}

int ExpGradDescent::predict(FeatureVec feature_vec) 
{
    int label;
    double val = std::inner_product(feature_vec.begin(), feature_vec.end(), weights_.begin(), 0.0);
    if (val > 0)
        label = 1;
    else
        label = -1;
    return label;
}

void ExpGradDescent::pushData(FeatureVec feature_vec,  int label) 
{
    int predicted_label = predict(feature_vec);
    if (predicted_label == label)
        return;
    else {
        double z = 0;
        for (size_t i = 0; i < weights_.size(); ++i) {
            double grad = -label*feature_vec[i];
            double arg = -learning_rate_*grad;
            weights_[i] = weights_[i]*exp(arg);
            z += weights_[i];
        }
        std::transform(weights_.begin(), weights_.end(), weights_.begin(),
                       std::bind1st(std::multiplies<double>(),1/z));
    }
}
