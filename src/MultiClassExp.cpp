#include <math.h>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <numeric>

#include <ol/MultiClassExp.h>

using namespace ol;

MultiClassExp::MultiClassExp(int num_rounds) : U_(10),
					       G_(1),
					       margin_(1)
{
    weights_plus_ = std::vector<std::vector<double> >(NUM_CLASSES, 
						      std::vector<double>(NUM_FEATURES, 0.5*U_/static_cast<double>(NUM_FEATURES)));
    weights_minus_ = std::vector<std::vector<double> >(NUM_CLASSES, 
						       std::vector<double>(NUM_FEATURES, 0.5*U_/static_cast<double>(NUM_FEATURES)));
    learning_rate_ = sqrt(std::log(NUM_FEATURES)/num_rounds)/G_;
}

Label MultiClassExp::predict(const FeatureVec& feature_vec) 
{
    double max_confidence = std::numeric_limits<double>::min();
    int best_class = -1;
    for (size_t i = 0; i < NUM_CLASSES; i++) {
	double confidence = getConfidence(feature_vec, weights_plus_[i], weights_minus_[i]);
	if (confidence > max_confidence) {
	    max_confidence = confidence;
	    best_class = i;
	}
    }
    return static_cast<Label>(best_class);
}

void MultiClassExp::pushData(const FeatureVec& feature_vec, Label label) 
{
    for (size_t i = 0; i < NUM_CLASSES; i++) {
	if (label == i)
	    continue;
	double confidence_correct = getConfidence(feature_vec, weights_plus_[label], weights_minus_[label]);
	double confidence_incorrect = getConfidence(feature_vec, weights_plus_[i], weights_minus_[i]);
	// if not correct by a margin, apply adjustment
	if (confidence_correct < confidence_incorrect + margin_) {
	    for (size_t j = 0; j < NUM_FEATURES; ++j) {
		double grad = -feature_vec[j];
		// update correct label weight
		weights_plus_[label][j] = weights_plus_[label][j]*exp(-learning_rate_*grad);
		weights_minus_[label][j] = weights_minus_[label][j]*exp(learning_rate_*grad);
		// update incorrect label weight
		weights_plus_[i][j] = weights_plus_[i][j]*exp(learning_rate_*grad);
		weights_minus_[i][j] = weights_minus_[i][j]*exp(-learning_rate_*grad);
	    }
	}
    }
    
    // normalize weights
    std::vector<double> weights_sum(NUM_CLASSES, 0);
    for (size_t i = 0; i < NUM_CLASSES; ++i) 
	for (size_t j = 0; j < NUM_FEATURES; ++j) 
	    weights_sum[i] += weights_plus_[i][j]+weights_minus_[i][j];

    for (size_t i = 0; i < NUM_CLASSES; ++i) {
	std::transform(weights_plus_[i].begin(), weights_plus_[i].end(), weights_plus_[i].begin(),
		       std::bind1st(std::multiplies<double>(),U_/weights_sum[i]));
	std::transform(weights_minus_[i].begin(), weights_minus_[i].end(), weights_minus_[i].begin(),
		       std::bind1st(std::multiplies<double>(),U_/weights_sum[i]));
    }
}

double MultiClassExp::getConfidence(const FeatureVec& feature_vec, const std::vector<double>& weights_plus, const std::vector<double>& weights_minus) 
{
    double confidence_plus = std::inner_product(feature_vec.begin(), feature_vec.end(),
						weights_plus.begin(), 0.0);
    double confidence_minus = std::inner_product(feature_vec.begin(), feature_vec.end(),
						 weights_minus.begin(), 0.0);
    double confidence = confidence_plus-confidence_minus;
    return confidence;
}
