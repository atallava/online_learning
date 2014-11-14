#pragma once
#include <vector>
#include <ol/Constants.h>
#include <ol/Dataset.h>

namespace ol {

    struct MultiClassPredictorParams {
        int num_rounds;
        double lambda;  // regularization
        // add more params here for the individual implementations
    };

    class MultiClassPredictor {
    public:
	MultiClassPredictor();
	virtual Label predict(const FeatureVec& feature_vec) = 0;
	virtual void pushData(const FeatureVec& feature_vec, Label label) = 0;
	void updateStreamLogs(Label true_label, Label predicted_label);
	void printStreamLogs();
    protected:
	int stream_rounds_;
	std::vector<double> stream_label_count_;
	std::vector<std::vector<double> > stream_confusion_matrix_;
    };

}
