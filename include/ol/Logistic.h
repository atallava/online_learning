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
    public:
        // uniform prior for the weights
        Logistic(int num_rounds, double lambda);
        int predict(const FeatureVec& feature_vec, double& confidence);
        void pushData(const FeatureVec& feature_vec,  int label);
        static std::pair<int,int> getBinaryLabels();
    private:
        int num_rounds_;
        std::vector<double> weights_;
        double learning_rate_;
	double lambda_;
        int current_iteration_;
    };
}
