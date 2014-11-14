#pragma once

#include <vector>

#include <ol/MultiClassPredictor.h>
#include <ol/Constants.h>

namespace ol {
    /**
     * @brief implements MultiClassLogistic regression for classification
     * @details assumes that the classes are 1 and 0
     * resource : http://people.cs.pitt.edu/~milos/courses/cs2710/lectures/Class22.pdf
     */
    class MultiClassLogistic : public MultiClassPredictor {
    public:
        // uniform prior for the weights
        MultiClassLogistic(MultiClassPredictorParams params);
        MultiClassLogistic(int num_rounds, double lambda);
        Label predict(const FeatureVec& feature_vec);
        void pushData(const FeatureVec& feature_vec, Label label);
    private:
        int num_rounds_;
        std::vector<std::vector<double> > weights_;
        double learning_rate_;
        int current_iteration_;
        double lambda_;

        double getNormalizer(const FeatureVec& x);
    };
}
