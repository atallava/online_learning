#pragma once
#include <vector>

namespace ol {
    class Predictor {
    public:
        virtual int predict(std::vector<double> features) = 0;
        virtual pushData(std::vector<double> features,  int label) = 0;
    };
}

