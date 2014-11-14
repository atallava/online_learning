#include <string>
#include <ol/CrossValidator.h>

using namespace ol;

int main(int argc, char** argv) {
    // std::string train_file_name ("data/oakland_part3_am_rf.node_features");
    // std::string test_file_name ("data/oakland_part3_an_rf.node_features");
    // {
    //     CrossValidatorParams params;

    //     // cross validate logistic regression
    //     params.predictor_type = std::string("multiexp");
    //     params.train_file_name = std::string("data/oakland_part3_am_rf.node_features");
    //     params.num_folds = 5;
    //     params.num_training_passes = 1;

    //     params.regularization.lower_limit = 1;
    //     params.regularization.upper_limit = 1e3;
    //     params.regularization.num_points = 1;

    //     CrossValidator cross_validator(params);
    //     auto best_param = cross_validator.getBestParameter();
        
    //     std::cout << std::string(100,'=') << std::endl << std::endl;
    //     printf("Best param for multiexp : %f\n", best_param.lambda);
    //     std::cout << std::string(100,'=') << std::endl << std::endl;
    // }
    // {
    //     CrossValidatorParams params;

    //     // cross validate logistic regression
    //     params.predictor_type = std::string("multilog");
    //     params.train_file_name = std::string("data/oakland_part3_am_rf.node_features");
    //     params.num_folds = 5;
    //     params.num_training_passes = 1;

    //     params.regularization.lower_limit = 1e-8;
    //     params.regularization.upper_limit = 1e1;
    //     params.regularization.num_points = 1;

    //     CrossValidator cross_validator(params);
    //     auto best_param = cross_validator.getBestParameter();

    //     std::cout << std::string(100,'=') << std::endl << std::endl;
    //     printf("Best param for multilog : %f\n", best_param.lambda);
    //     std::cout << std::string(100,'=') << std::endl << std::endl;
    // }
    // {
    //     CrossValidatorParams params;

    //     // cross validate logistic regression
    //     params.predictor_type = std::string("svm");
    //     params.train_file_name = std::string("data/oakland_part3_am_rf.node_features");
    //     params.num_folds = 5;
    //     params.num_training_passes = 1;

    //     params.regularization.lower_limit = 1e-9;
    //     params.regularization.upper_limit = 1e-5;
    //     params.regularization.num_points = 1;

    //     CrossValidator cross_validator(params);
    //     auto best_param = cross_validator.getBestParameter();

    //     std::cout << std::string(100,'=') << std::endl << std::endl;
    //     printf("Best param for svm : %f\n", best_param.lambda);
    //     std::cout << std::string(100,'=') << std::endl << std::endl;
    // }
    {
        CrossValidatorParams params;

        // cross validate logistic regression
        params.predictor_type = std::string("kernel_svm");
        params.train_file_name = std::string("data/oakland_part3_am_rf.node_features");
        params.num_folds = 5;
        params.num_training_passes = 1;

        params.regularization.lower_limit = 1e-12;
        params.regularization.upper_limit = 1e-5;
        params.regularization.num_points = 1;

        CrossValidator cross_validator(params);
        auto best_param = cross_validator.getBestParameter();

        std::cout << std::string(100,'=') << std::endl << std::endl;
        printf("Best param for kernel_svm : %f\n", best_param.lambda);
        std::cout << std::string(100,'=') << std::endl << std::endl;
    }

    return 0;
}
