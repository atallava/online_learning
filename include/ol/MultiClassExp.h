#pragma once
#include <vector>

#include <ol/MultiClassPredictor.h>
#include <ol/Dataset.h>

namespace ol {
class MultiClassExp : public MultiClassPredictor {
    public:
    MultiClassExp(MultiClassPredictorParams params);
	MultiClassExp(int num_rounds, double U);
	Label predict(const FeatureVec& feature_vec);
	void pushData(const FeatureVec& feature_vec, Label label);
	double getConfidence(const FeatureVec& feature_vec, const std::vector<double>& weights_plus, const std::vector<double>& weights_minus);
	void printWeights();
    private:
	std::vector<std::vector<double> > weights_plus_;
	std::vector<std::vector<double> > weights_minus_;
	double U_;
	double G_;
	double learning_rate_;
	double margin_;
    };
}
