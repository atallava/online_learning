#include <math.h>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ol/ExpGradDescent.h>
#include <ol/Constants.h>

using namespace ol;

ExpGradDescent::ExpGradDescent(int num_rounds)
{
    weights_ = std::vector<double>(NUM_FEATURES, 1/static_cast<double>(NUM_FEATURES));
    double G_ = 1;
    learning_rate_ = sqrt(std::log(NUM_FEATURES)/num_rounds)/G_;
}

int ExpGradDescent::predict(const FeatureVec& feature_vec, double& confidence) 
{
    confidence = std::inner_product(feature_vec.begin(), feature_vec.end(),
				    weights_.begin(), 0.0);
    return (confidence > 0) ? 1 : -1;
}

void ExpGradDescent::pushData(const FeatureVec& feature_vec,  int label) 
{
    // hinge part : if predicted label is fine, don't update
    double confidence;
    int predicted_label = predict(feature_vec, confidence);
    if (label*confidence >= 1) {
        return;
    }

    // Hacked-up solution for negative weights
    else {
        for (size_t i = 0; i < weights_.size(); ++i) {
	    int sign_weight = (weights_[i] > 0) ? 1 : -1;
	    double grad = -label*feature_vec[i];
	    double arg = -sign_weight*learning_rate_*grad;
	    
	    double dummy_weight = weights_[i]-learning_rate_*grad;
	    int sign_dummy = (dummy_weight > 0) ? 1 : -1;

            weights_[i] = weights_[i]*exp(arg);
	    
	    if (sign_dummy != sign_weight) 
		weights_[i] = -weights_[i];
        }

	double norm = 0;
	for (size_t i = 0; i < weights_.size(); ++i)
	    norm += weights_[i]*weights_[i];
	norm = sqrt(norm);
	if (norm > 1)
	    std::transform(weights_.begin(), weights_.end(), weights_.begin(),
			   std::bind1st(std::multiplies<double>(),1/norm));
    }
}

