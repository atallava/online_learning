#pragma once
#include <string>

namespace ol {
    class Validator {
    public:
        double validateOnDataset(std::string file_name, std::string predictor_type);
    private:
        int num_train_ = 40000;
        int num_test_ = 20000;
    };
}
