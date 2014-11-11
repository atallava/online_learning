#include <stdexcept>
#include <limits>

#include <ol/OneVsAll.h>

using namespace ol;

OneVsAll::OneVsAll(int num_rounds, std::string type)
{
    // initialize the predictors based on the type
    if (type.compare(std::string("logistic")) == 0){
        for (int i = 0; i < NUM_CLASSES; i++)
            binary_predictors_.push_back(std::make_shared<Logistic>(num_rounds));
        binary_labels_ = Logistic::getBinaryLabels();
    } else if (type.compare(std::string("exp")) == 0) {
        for (int i = 0; i < NUM_CLASSES; i++)
            binary_predictors_.push_back(std::make_shared<ExpGradDescent>(num_rounds));
        binary_labels_ = ExpGradDescent::getBinaryLabels();
    } else {
        throw std::runtime_error("Can't find predictor");
    }
}

void OneVsAll::pushData(const FeatureVec& feature_vec, Label label)
{
    for(int i = 0; i < NUM_CLASSES; i++) {
        int y = (static_cast<int>(label) == i) ?
                            binary_labels_.second:binary_labels_.first;
        binary_predictors_[i]->pushData(feature_vec, y);
    }
}


Label OneVsAll::predict(const FeatureVec& feature_vec)
{
    int max_idx = -1;
    double max_confidence = static_cast<double>(std::numeric_limits<int>::min());
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        double this_confidence = 0.0;
        int predicted_y = binary_predictors_[i]->predict(feature_vec, this_confidence);
        if (this_confidence > max_confidence) {
            max_confidence = this_confidence;
            max_idx = i;
        }
    }
    return static_cast<Label>(max_idx);
}
