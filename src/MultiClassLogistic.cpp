#include <numeric>

#include <ol/MultiClassLogistic.h>

using namespace ol;

MultiClassLogistic::MultiClassLogistic(MultiClassPredictorParams params)
    :   num_rounds_(params.num_rounds),
        weights_(NUM_CLASSES - 1),  // one less because it is normalized
        current_iteration_(0),
        lambda_(params.lambda)
{
    std::for_each(weights_.begin(), weights_.end(), [](std::vector<double>& w){
        w.resize(NUM_FEATURES, 1/NUM_FEATURES);
    });
}

MultiClassLogistic::MultiClassLogistic(int num_rounds, double lambda)
    :   num_rounds_(num_rounds),
        weights_(NUM_CLASSES - 1),  // one less because it is normalized
        current_iteration_(0),
        lambda_(lambda)
{
    std::for_each(weights_.begin(), weights_.end(), [](std::vector<double>& w){
        w.resize(NUM_FEATURES, 1/NUM_FEATURES);
    });
}


void MultiClassLogistic::pushData(const FeatureVec& features, Label label)
{
    current_iteration_++;
    Label predicted_label = predict(features);
    updateStreamLogs(label, predicted_label);

    // set the learning rate adaptively
    learning_rate_ = static_cast<double>(1.0/std::sqrt(current_iteration_));

    //regularization
    for(unsigned int i=0; i < weights_.size(); i++)
        for(unsigned int j=0; j < weights_[i].size(); j++)
            weights_[i][j] -= learning_rate_ * lambda_ * weights_[i][j];

    double z = getNormalizer(features);
    for (size_t i = 0; i < weights_.size(); i++) {
        double wx = std::inner_product(weights_[i].begin(),
                                weights_[i].end(), features.begin(), 0.0);
        for (size_t j = 0; j < weights_[i].size(); j++) {
            weights_[i][j] -= learning_rate_*(std::exp(wx)/z - 
                static_cast<double>(label == i))*features[j];
        }
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
