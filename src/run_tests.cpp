#include <ol/Test.h>

using namespace ol;

int main() {
    Test t = Test();
    //t.testVizPCD("data/am.pcd");
    //t.testDataset("data/oakland_part3_am_rf.node_features");
    //t.testVizPoints();
    t.validatePredictor("data/oakland_part3_am_rf.node_features","logistic");
    // t.validatePredictor("data/oakland_part3_am_rf.node_features","expgraddescent");
}
