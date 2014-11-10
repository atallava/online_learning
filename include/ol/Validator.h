#pragma once
#include <string>

namespace ol {
    class Validator {
    public:
        Validator() : num_train_(70000), num_test_(19822) {}
        double validateOnDataset(std::string file_name, std::string predictor_type);
    private:
        size_t num_train_;
        size_t num_test_;
    };
}
