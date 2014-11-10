#pragma once
#include <string>

namespace ol {
    class Validator {
    public:
        Validator() : num_train_(40000), num_test_(10) {}
        double validateOnDataset(std::string file_name, std::string predictor_type);
    private:
        size_t num_train_;
        size_t num_test_;
    };
}
