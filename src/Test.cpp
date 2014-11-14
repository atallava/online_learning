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
    std::string file_name = "data/oakland_part3_an_rf.node_features";
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

bool Test::validatePredictor(std::string file_name, std::string predictor_type, 
			     double predictor_param,
                             bool adjust_for_under_represented_classes,
                             int num_training_passes)
{
    Validator v;
    bool print_choice = true;
    double accuracy = v.validate(file_name, predictor_type, predictor_param, print_choice,
				 adjust_for_under_represented_classes, num_training_passes);
    return true;
}

bool Test::validatePredictor(std::string train_file_name, std::string test_file_name,
			     std::string predictor_type, double predictor_param,
			     bool adjust_for_under_represented_classes,
			     int num_training_passes)
{
    Validator v;
    bool print_choice = true;
    bool viz_choice = true;
    double accuracy = v.validate(train_file_name, test_file_name, predictor_type, predictor_param, print_choice,
                                 viz_choice, adjust_for_under_represented_classes, num_training_passes);
    return true;
}
