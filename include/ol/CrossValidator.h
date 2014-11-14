#pragma once
#include <string>
#include <memory>

#include <ol/Validator.h>
#include <ol/Dataset.h>


namespace ol {
    struct HyperParamSettings {
        double lower_limit;
        double upper_limit;
        int num_points;
    };

    struct CrossValidatorParams {
        std::string predictor_type;
        std::string train_file_name;
        int num_folds;
        int num_training_passes;

        // params to tweak
        HyperParamSettings regularization;

        ValidatorParams getValidatorParams();
    };

    class CrossValidator {
    public:
        CrossValidator(CrossValidatorParams params);
        ValidatorParams getBestParameter();
        double averageAccuracyForParameters(ValidatorParams params);
    private:
        std::pair<int,int> getTestFoldIndices(int fold_id);
        CrossValidatorParams params_;
        std::unique_ptr<Validator> validator_;
        Dataset train_dataset_;
    };
}
