#include <string>
#include <ol/Test.h>

using namespace ol;

int main() {
    Test t = Test();
    //t.testVizPCD("data/am.pcd");
    //t.testDataset("data/oakland_part3_am_rf.node_features");
    //t.testVizPoints();

    std::string train_file_name ("data/oakland_part3_am_rf.node_features");
    std::string test_file_name ("data/oakland_part3_an_rf.node_features");


    //Best Params
    //method,   adjust_for_under_represented_classes, num_training_passes
    //Logistic, false, 3
    //Exp,      true/ false, 2
    //SVM,      true,  4

    //t.validatePredictor(train_file_name, test_file_name, "logistic", false, 3); 
    //t.validatePredictor(train_file_name, test_file_name, "exp", false, 3);
    t.validatePredictor(train_file_name, test_file_name, "multiexp", true, 1);
    t.validatePredictor(train_file_name, test_file_name, "svm", true, 4);
}
