#pragma once
#include <string>
#include <vector>

#include <ol/Dataset.h>

namespace ol {
    class Validator {
    public:
        Validator() : single_scene_num_train_(10), single_scene_num_test_(1) {}
	// use subsets of one scene for train and test
	double validate(std::string file_name, std::string predictor_type, bool print_choice,
			bool adjust_for_under_represented_classes,
			int num_training_passes);
	// use one scene as train and another as test
	double validate(std::string train_file_name, std::string test_file_name, 
			std::string predictor_type, bool print_choice,
			bool adjust_for_under_represented_classes,
			int num_training_passes);
	// workhorse
	double validate(std::vector<FeatureVec> train_feature_vecs, std::vector<Label> train_labels, 
			std::vector<FeatureVec> test_feature_vecs, std::vector<Label> test_labels, 
			std::string predictor_type, bool print_choice,
			bool adjust_for_under_represented_classes,
			int num_training_passes);
    private:
        size_t single_scene_num_train_;
        size_t single_scene_num_test_;
    };
}
