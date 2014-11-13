#include <math.h>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ol/ExpGradDescent.h>
#include <ol/Constants.h>

using namespace ol;

ExpGradDescent::ExpGradDescent(int num_rounds, double U) : U_(U),
						 G_(1)
{
    weights_plus_ = std::vector<double>(NUM_FEATURES, 0.5*U_/static_cast<double>(NUM_FEATURES));
    weights_minus_ = std::vector<double>(NUM_FEATURES, 0.5*U_/static_cast<double>(NUM_FEATURES));
    learning_rate_ = sqrt(std::log(NUM_FEATURES)/num_rounds)/G_;
}

int ExpGradDescent::predict(const FeatureVec& feature_vec, double& confidence) 
{
    double confidence_plus = std::inner_product(feature_vec.begin(), feature_vec.end(),
						weights_plus_.begin(), 0.0);
    double confidence_minus = std::inner_product(feature_vec.begin(), feature_vec.end(),
						   weights_minus_.begin(), 0.0);
    confidence = confidence_plus-confidence_minus;
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

    else {
        for (size_t i = 0; i < weights_plus_.size(); ++i) {
	    double grad = -label*feature_vec[i];
	    
            weights_plus_[i] = weights_plus_[i]*exp(-learning_rate_*grad);
	    weights_minus_[i] = weights_minus_[i]*exp(learning_rate_*grad);
        }
	
	double weights_sum = 0;
	for (size_t i = 0; i < weights_plus_.size(); ++i)
	    weights_sum += weights_plus_[i]+weights_minus_[i];
	
	// normalize weights
	std::transform(weights_plus_.begin(), weights_plus_.end(), weights_plus_.begin(),
			   std::bind1st(std::multiplies<double>(),U_/weights_sum));
	std::transform(weights_minus_.begin(), weights_minus_.end(), weights_minus_.begin(),
		       std::bind1st(std::multiplies<double>(),U_/weights_sum));
    }
}

void ExpGradDescent::printWeights() {
    std::cout << "Positive weights: " << std::endl;
    for (size_t i = 0; i < weights_plus_.size(); ++i) 
	std::cout << std::left << std::setw(20) << weights_plus_[i];
    std:: cout << std::endl;
    std::cout << "Negative weights: " << std::endl;
    for (size_t i = 0; i < weights_minus_.size(); ++i) 
	std::cout << std::left << std::setw(20) << weights_minus_[i];
}
