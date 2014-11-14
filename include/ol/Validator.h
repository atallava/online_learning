#pragma once
#include <string>
#include <vector>

#include <ol/MultiClassPredictor.h>
#include <ol/Dataset.h>
#include <ol/Visualizer.h>

namespace ol {
    
    struct ValidatorParams{
        int num_training_passes;
        double lambda;  // regularization parameter
        MultiClassPredictorParams getPredictorParams();
    };

    class Validator {
    public:
        Validator() : single_scene_num_train_(10), single_scene_num_test_(1) {}
	// use subsets of one scene for train and test
	double validate(std::string file_name, std::string predictor_type, double predictor_param,
			bool print_choice, 
			bool adjust_for_under_represented_classes,
			int num_training_passes);
	// use one scene as train and another as test
	double validate(std::string train_file_name, std::string test_file_name, 
			std::string predictor_type, double predictor_param,
			bool print_choice, bool viz_choice,
			bool adjust_for_under_represented_classes,
			int num_training_passes);
	// workhorse
	double validate(std::vector<FeatureVec> train_feature_vecs, std::vector<Label> train_labels, 
			std::vector<FeatureVec> test_feature_vecs, std::vector<Label> test_labels, 
			std::string predictor_type, double predictor_param, bool print_choice,
			bool adjust_for_under_represented_classes,
			int num_training_passes);
	std::vector<Label> getPredictedLabels(const std::vector<FeatureVec>& feature_vec,
					      MultiClassPredictor* mcp);
	MultiClassPredictor* trainPredictor(std::vector<FeatureVec> train_feature_vecs, 
					    std::vector<Label> train_labels, 
					    std::string predictor_type, double predictor_param,
					    bool adjust_for_under_represented_classes, 
					    int num_training_passes);
	double testPredictor(std::vector<FeatureVec> test_feature_vecs,
			     std::vector<Label> test_labels,
			     MultiClassPredictor* mcp, bool print_choice);

    // added for the cross validation features
    void trainPredictor(const Dataset& dataset, std::pair<int,int> testset, std::string predictor_type, ValidatorParams params);
    double testPredictor(const Dataset& dataset, std::pair<int,int> testset);


    private:
        size_t single_scene_num_train_;
        size_t single_scene_num_test_;
        std::shared_ptr<MultiClassPredictor> predictor_;
    };
}
