#include <string>
#include <ol/CrossValidator.h>

using namespace ol;

int main(int argc, char** argv) {
    // std::string train_file_name ("data/oakland_part3_am_rf.node_features");
    // std::string test_file_name ("data/oakland_part3_an_rf.node_features");

    CrossValidatorParams params;

    // cross validate logistic regression
    params.predictor_type = std::string("multilog");
    params.train_file_name = std::string("data/oakland_part3_am_rf.node_features");
    params.num_folds = 10;
    params.num_training_passes = 3;

    params.regularization.lower_limit = 1e-4;
    params.regularization.upper_limit = 1e-3;
    params.regularization.num_points = 10;

    CrossValidator cross_validator(params);
    auto best_param = cross_validator.getBestParameter();

    printf("Best param for multilog : %f\n", best_param.lambda);

    //Best Params
    //method,   adjust_for_under_represented_classes, num_training_passes, param
    //Logistic, false, 3, ?
    //Exp,      true/ false, 2, ?
    //MultiExp, false, 4, U_ = 5
    //SVM,      true,  4, lambda_ = 0.0001
    //MultiLog,      true,  3, lambda_ = 0.0001
    //KernelSVM, ?, ?, ?

    // t.validatePredictor(train_file_name, test_file_name, "logistic", 0.0001, false, 3); 
    //t.validatePredictor(train_file_name, test_file_name, "multilog", 0.0001, true, 4); 

    //t.validatePredictor(train_file_name, test_file_name, "exp", 5, false, 3);
    // t.validatePredictor(train_file_name, test_file_name, "multiexp", 5, true, 4);

    //t.validatePredictor(train_file_name, test_file_name, "svm", 0.0001, true, 4);
    //t.validatePredictor(train_file_name, test_file_name, "kernel_svm", 0.0001, true, 1);
    return 0;
}
