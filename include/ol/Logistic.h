#pragma once

#include <vector>

#include <ol/Predictor.h>
#include <ol/Constants.h>

namespace ol {
    /**
     * @brief implements logistic regression for classification
     * @details assumes that the classes are 1 and 0
     * resource : http://people.cs.pitt.edu/~milos/courses/cs2710/lectures/Class22.pdf
     */
    class Logistic : public Predictor {
        // uniform prior for the weights
        Logistic();
        int predict(const FeatureVec& feature_vec);
        void pushData(const FeatureVec& feature_vec,  int label);
        static std::pair<int,int> getBinaryLabels();
    private:
        std::vector<double> weights_;
        double learning_rate_;
        int current_iteration_;
    };
}