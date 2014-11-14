#include <string>
#include <ol/Test.h>

using namespace ol;

int main() {
    Test t = Test();
    //t.testVizPCD("data/am.pcd");
    //t.testVizPoints();

    std::string train_file_name ("data/oakland_part3_am_rf.node_features");
    std::string test_file_name ("data/oakland_part3_an_rf.node_features");


    //Best Params
    //method,   adjust_for_under_represented_classes, num_training_passes, param
    //Logistic, false, 3, ?
    //Exp,      true/ false, 2, ?
    //MultiExp, false, 4, U_ = 5
    //SVM,      true,  4, lambda_ = 0.0001
    //MultiLog,      true,  3, lambda_ = 0.0001
    //KernelSVM, ?, ?, ?

    // t.validatePredictor(train_file_name, test_file_name, "logistic", 0.0001, false, 3); 
    t.validatePredictor(train_file_name, test_file_name, "logistic", 0.0001, false, 3); 
    t.validatePredictor(train_file_name, test_file_name, "multilog", 0.0001, true, 4); 

    t.validatePredictor(train_file_name, test_file_name, "exp", 5, false, 3);
    t.validatePredictor(train_file_name, test_file_name, "multiexp", 5, true, 4);

    t.validatePredictor(train_file_name, test_file_name, "svm", 0.0001, true, 4);
    t.validatePredictor(train_file_name, test_file_name, "kernel_svm", 0.000000000001, true, 1);

    //t.validatePredictor(train_file_name, test_file_name, "kernel_svm", 0.0001, true, 1);
}
