#include <ol/Test.h>

#include <ol/Visualizer.h>

using namespace ol;

bool Test::testVizPCD(std::string file_name)
{
    Visualizer viz;
    viz.visualize(file_name);
    return true;
}
