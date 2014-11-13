#pragma once
#include <string>

#include <ol/Validator.h>

namespace ol {
    class CrossValidator {
    public:
        CrossValidator() : validator_(), num_folds_(10) {}
	double getBestParameter();
	double accuracyForParameter(double param);
    private:
	Validator validator_;
	int num_folds_;
	std::string predictor_type_;
    };
}
