#pragma once
#include <string>
#include <vector>

#include <ol/Dataset.h>

namespace ol {
    class Validator {
    public:
        Validator() : single_scene_num_train_(40000), single_scene_num_test_(100) {}
	// use subsets of one scene for train and test
	double validate(std::string file_name, std::string predictor_type, bool print_choice = false);
	// use one scene as train and another as test
	double validate(std::string train_file_name, std::string test_file_name, 
			std::string predictor_type, bool print_choice = false);
	// workhorse
	double validate(std::vector<FeatureVec> train_feature_vecs, std::vector<Label> train_labels, 
			std::vector<FeatureVec> test_feature_vecs, std::vector<Label> test_labels, 
			std::string predictor_type, bool print_choice = false);
    private:
        size_t single_scene_num_train_;
        size_t single_scene_num_test_;
    };
}
