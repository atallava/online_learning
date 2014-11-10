#include <ol/Test.h>

#include <ol/Visualizer.h>
#include <ol/Validator.h>
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

bool Test::validatePredictor(std::string file_name, std::string predictor_type)
{
    Validator v;
    bool print_choice = true;
    double accuracy = v.validate(file_name, predictor_type, print_choice);
    return true;
}

bool Test::validatePredictor(std::string train_file_name, std::string test_file_name,
			       std::string predictor_type)
{
    Validator v;
    bool print_choice = true;
    double accuracy = v.validate(train_file_name, test_file_name, predictor_type, print_choice);
    return true;
}
