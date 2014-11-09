#include <ol/Test.h>

#include <ol/Visualizer.h>
#include <ol/Dataset.h>

using namespace ol;

bool Test::testVizPCD(std::string file_name)
{
    Visualizer viz;
    viz.visualize(file_name);
    return true;
}

bool Test::testVizPoints()
{
    std::string file_name = "data/oakland_part3_am_rf.node_features";
    Dataset d(file_name);
    Visualizer viz;
    viz.visualize(d.points(),d.labels());
    return true;
}

bool Test::testDataset(std::string file_name)
{
    Dataset d(file_name);
    return true;
}
