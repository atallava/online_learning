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
    //t.validatePredictor(train_file_name, test_file_name, "logistic"); 
    t.validatePredictor(train_file_name, test_file_name, "expgraddescent");
}
