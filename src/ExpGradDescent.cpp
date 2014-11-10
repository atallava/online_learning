#include <math.h>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ol/ExpGradDescent.h>

using namespace ol;

ExpGradDescent::ExpGradDescent(int num_rounds) 
{
    weights_ = std::vector<double>(NUM_FEATURES, 1/NUM_FEATURES);
    double G_ = 1;
    learning_rate_ = sqrt(std::log(NUM_FEATURES)/num_rounds)/G_;
}

int ExpGradDescent::predict(const FeatureVec& feature_vec, double& confidence) 
{
    int label;
    confidence = std::inner_product(feature_vec.begin(), feature_vec.end(),
        weights_.begin(), 0.0);
    return (confidence > 0) ? 1 : -1;
}

void ExpGradDescent::pushData(const FeatureVec& feature_vec,  int label) 
{
    // hinge part : if predicted label is fine, don't update
    double confidence;
    int predicted_label = predict(feature_vec, confidence);
    if (confidence >= 1)
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
