#include <numeric>

#include <ol/MultiClassLogistic.h>

using namespace ol;

MultiClassLogistic::MultiClassLogistic(int num_rounds)
    :   num_rounds_(num_rounds),
        weights_(NUM_CLASSES - 1),  // one less because it is normalized
        current_iteration_(0),
        lambda_(0.0001)
{
    std::for_each(weights_.begin(), weights_.end(), [](std::vector<double>& w){
        w.resize(NUM_FEATURES, 1/NUM_FEATURES);
    });
}


void MultiClassLogistic::pushData(const FeatureVec& features, Label label)
{
    current_iteration_++;
    // skip the last label because our vector is just 0s
    if (label == NUM_CLASSES - 1)
        return;
    // set the learning rate adaptively
    learning_rate_ = static_cast<double>(1.0/current_iteration_);

    //regularization - not for all. Just for the correct one.
    for(unsigned int j=0; j < weights_[label].size(); j++)
        weights_[label][j] -= learning_rate_ * lambda_ * weights_[label][j];

    double wx_correct = std::inner_product(weights_[label].begin(),
                                weights_[label].end(), features.begin(), 0.0);
    double z = getNormalizer(features);
    for (size_t j = 0; j < weights_[label].size(); j++) {
        weights_[label][j] -= learning_rate_*(std::exp(wx_correct)/z -1)*features[j];
    }
}

Label MultiClassLogistic::predict(const FeatureVec& features)
{
    double z = getNormalizer(features);
    std::vector<double> confidence_vec(NUM_CLASSES, 0.0);
    for (size_t i = 0; i < NUM_CLASSES-1; i++) {
        double wx = std::inner_product(weights_[i].begin(), weights_[i].end(),
            features.begin(), 0.0);
        confidence_vec[i] = std::exp(wx)/z;
    }
    confidence_vec[NUM_CLASSES - 1] = 1/z;
    
    // find the max and return the index
    auto it = std::max_element(confidence_vec.begin(), confidence_vec.end());
    return static_cast<Label>(it - confidence_vec.begin());
}

double MultiClassLogistic::getNormalizer(const FeatureVec& x) {
    double z = 1.0;
    for (auto& w : weights_) {
        double wx = std::inner_product(w.begin(), w.end(), x.begin(), 0.0);
        z += std::exp(wx);
    }
    return z;
}