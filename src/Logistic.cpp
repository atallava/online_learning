#include <numeric>

#include <ol/Logistic.h>

using namespace ol;

Logistic::Logistic(int num_rounds)
    :   num_rounds_(num_rounds),
        weights_(NUM_FEATURES, 1.0/static_cast<double>(NUM_FEATURES)),
        current_iteration_(0)
{}

std::pair<int,int> Logistic::getBinaryLabels()
{
    return std::make_pair(0,1);
}

void Logistic::pushData(const FeatureVec& features, int label)
{
    current_iteration_++;
    // set the learning rate adaptively
    learning_rate_ = static_cast<double>(1.0/current_iteration_);
    // compute w^T*x
    double w_t_x = std::inner_product(weights_.begin(), weights_.end(),
        features.begin(), 0);
    // update the weights
    for (size_t j = 0; j < weights_.size(); j++) {
        weights_[j] += learning_rate_ * (label - sigmoid(w_t_x))*features[j];
    }
}

int Logistic::predict(const FeatureVec& features, double& confidence)
{
    double w_t_x = std::inner_product(weights_.begin(), weights_.end(),
        features.begin(), 0);
    // get the probability
    confidence = sigmoid(w_t_x);
    return static_cast<int>(confidence > 0.5);
}
