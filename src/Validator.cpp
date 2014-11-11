#include <iomanip>
#include <ol/Validator.h>
#include <ol/Dataset.h>
#include <ol/OneVsAll.h>
#include <ol/MultiClassSVM.h>
#include <ol/Constants.h>

using namespace ol;

double Validator::validate(std::string file_name, std::string predictor_type, 
				    bool print_choice) 
{
    Dataset dset(file_name);
    std::vector<FeatureVec> feature_vecs = dset.feature_vecs();
    std::vector<Label> labels = dset.labels();

    // form train and test
    std::vector<FeatureVec> train_feature_vecs(feature_vecs.begin(), feature_vecs.begin()+single_scene_num_train_-1);
    std::vector<Label> train_labels(labels.begin(), labels.begin()+single_scene_num_train_-1);
    std::vector<FeatureVec> test_feature_vecs(feature_vecs.begin()+single_scene_num_train_,
					      feature_vecs.begin()+single_scene_num_train_+single_scene_num_test_-1);
    std::vector<Label> test_labels(labels.begin()+single_scene_num_train_, labels.begin()+single_scene_num_train_+single_scene_num_test_-1);

    return validate(train_feature_vecs, train_labels,
		    test_feature_vecs, test_labels,
		    predictor_type, print_choice);
}

double Validator::validate(std::string train_file_name, std::string test_file_name, 
				    std::string predictor_type, bool print_choice) 
{
    Dataset train_dset(train_file_name);
    Dataset test_dset(test_file_name);

    // form train and test
    std::vector<FeatureVec> train_feature_vecs = train_dset.feature_vecs();
    std::vector<Label> train_labels = train_dset.labels();
    std::vector<FeatureVec> test_feature_vecs = test_dset.feature_vecs();
    std::vector<Label> test_labels = test_dset.labels();

    return validate(train_feature_vecs, train_labels,
		    test_feature_vecs, test_labels,
		    predictor_type, print_choice);
}



double Validator::validate(std::vector<FeatureVec> train_feature_vecs, std::vector<Label> train_labels, 
			   std::vector<FeatureVec> test_feature_vecs, std::vector<Label> test_labels, 
			   std::string predictor_type, bool print_choice)
{
    size_t num_train = train_labels.size();
    size_t num_test = test_labels.size();

    // create predictor
    MultiClassPredictor* mcp;
    if(predictor_type.compare(std::string("svm")) == 0)
      mcp = new MultiClassSVM(num_train);
    else
      mcp = new OneVsAll(num_train, predictor_type);

    // train
    for (size_t i = 0; i < num_train; ++i) 
        mcp->pushData(train_feature_vecs[i], train_labels[i]);

    std::vector<double> test_label_count(NUM_CLASSES, 0);
    std::vector<double> test_label_freq(NUM_CLASSES, 0);
    std::vector<double> per_label_accuracy(NUM_CLASSES, 0);
    std::vector<std::vector<double> > confusion_matrix(NUM_CLASSES, std::vector<double>(NUM_CLASSES, 0));
    double accuracy = 0.0;

    // test
    Label predicted_label;
    for (size_t i = 0; i < num_test; ++i) {
        predicted_label = mcp->predict(test_feature_vecs[i]);
	// printf("actual : %d, predicted_label : %d\n", test_labels[i],
        //                                                   predicted_label);
	test_label_count[test_labels[i]] += 1;
	confusion_matrix[test_labels[i]][predicted_label] += 1;
	if (predicted_label == test_labels[i]) 
	    per_label_accuracy[test_labels[i]] += 1;
    }
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
	if (test_label_count[i] > 0) {
	    accuracy += per_label_accuracy[i];
	    per_label_accuracy[i] /= test_label_count[i];
	}
	test_label_freq[i] = test_label_count[i]/num_test;
    }
    accuracy /= num_test;

    // pretty printing
    if (print_choice) {
	std::cout << "Predictor: " << predictor_type << "\n\n";
	
	std::cout << std::left << std::setw(20) << "CLASS NAME" 
		  << std::left << std::setw(20) << "CLASS FREQUENCY" 
		  << std::left << std::setw(20) << "PER CLASS ACCURACY" << std::endl;

	for (size_t i = 0; i < NUM_CLASSES; ++i) {
	    std::cout << std::left << std::setw(20) << CLASS_NAMES[i] 
		      << std::left << std::setw(20) << test_label_freq[i] 
		      << std::left << std::setw(20) << per_label_accuracy[i] << std::endl;
	}
	std::cout << "\n";
	std::cout << "Confusion Matrix: \n";
	for (size_t i = 0; i < confusion_matrix.size(); ++i) {
	    for (size_t j = 0; j < confusion_matrix[0].size(); ++j) {
		std::cout << std::left << std::setw(10) << confusion_matrix[i][j];
	    }
	    std::cout << "\n\n";
	}

	printf("Overall accuracy: %.2f\n\n", accuracy);
    }

    delete mcp;

    return accuracy;
}
