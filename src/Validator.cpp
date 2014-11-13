#include <iomanip>
#include <ctime>
#include <ol/Validator.h>
#include <ol/Dataset.h>
#include <ol/OneVsAll.h>
#include <ol/MultiClassSVM.h>
#include <ol/MultiClassExp.h>
#include <ol/MultiClassLogistic.h>
#include <ol/Constants.h>

using namespace ol;

double Validator::validate(std::string file_name, std::string predictor_type, 
			   bool print_choice, 
			   bool adjust_for_under_represented_classes,
			   int num_training_passes)
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
		    predictor_type, print_choice,
		    adjust_for_under_represented_classes,
		    num_training_passes);
}

double Validator::validate(std::string train_file_name, std::string test_file_name, 
			   std::string predictor_type, bool print_choice, bool viz_choice,
			   bool adjust_for_under_represented_classes,
			   int num_training_passes)
{
    Dataset train_dset(train_file_name);
    //train_dset.whitenData();
    train_dset.shuffleData();
    Dataset test_dset(test_file_name);

    // form train and test
    std::vector<FeatureVec> train_feature_vecs = train_dset.feature_vecs();
    std::vector<Label> train_labels = train_dset.labels();
    std::vector<FeatureVec> test_feature_vecs = test_dset.feature_vecs();
    std::vector<Label> test_labels = test_dset.labels();
    
    std::clock_t begin = std::clock();    
    MultiClassPredictor* mcp = trainPredictor(train_feature_vecs, train_labels, predictor_type, 
					      adjust_for_under_represented_classes, num_training_passes);
    std::clock_t end = std::clock();

    if (print_choice) {
	std::cout << "Predictor: " << predictor_type << "\n\n";
	double elapsed_time = double(end-begin)/CLOCKS_PER_SEC;
	printf("Training time (CPU): %0.2fs\n\n", elapsed_time);
	mcp->printStreamLogs();
	std::cout << std::string(50,'-') << std::endl;
    }

    // visualize train pcd
    // if (viz_choice) {
    // 	std::vector<Label> predicted_labels = getPredictedLabels(train_feature_vecs, mcp);
    // 	Visualizer vizer_train;
    // 	vizer_train.visualize(train_dset.points(), train_labels,
    // 			train_dset.points(), predicted_labels);
    // }

    double accuracy = testPredictor(test_feature_vecs, test_labels, mcp, print_choice);

    // visualize test pcd
    if (viz_choice) {
	std::vector<Label> predicted_labels = getPredictedLabels(test_feature_vecs, mcp);
	Visualizer vizer;
	vizer.visualize(test_dset.points(), test_labels,
			test_dset.points(), predicted_labels);
    }

    delete mcp;
    return accuracy;
}


double Validator::validate(std::vector<FeatureVec> train_feature_vecs, std::vector<Label> train_labels, 
			   std::vector<FeatureVec> test_feature_vecs, std::vector<Label> test_labels, 
			   std::string predictor_type, bool print_choice,
			   bool adjust_for_under_represented_classes,
			   int num_training_passes)
{
    MultiClassPredictor* mcp = trainPredictor(train_feature_vecs, train_labels, predictor_type, 
					      adjust_for_under_represented_classes, num_training_passes);
    if (print_choice)
	std::cout << "Predictor: " << predictor_type << "\n\n";

    double accuracy = testPredictor(test_feature_vecs, test_labels, mcp, print_choice);

    delete mcp;
    return accuracy;
}

std::vector<Label> Validator::getPredictedLabels(const std::vector<FeatureVec>& feature_vecs,
				      MultiClassPredictor* mcp)
{
    std::vector<Label> predicted_labels(feature_vecs.size());
    for (size_t i = 0; i < feature_vecs.size(); ++i) 
	predicted_labels[i] = mcp->predict(feature_vecs[i]);

    return predicted_labels;
}

MultiClassPredictor* Validator::trainPredictor(std::vector<FeatureVec> train_feature_vecs, 
				    std::vector<Label> train_labels, 
				    std::string predictor_type,
				    bool adjust_for_under_represented_classes, 
				    int num_training_passes)
{
    size_t num_train = train_labels.size();

    // create predictor
    MultiClassPredictor* mcp;
    if (predictor_type.compare(std::string("svm")) == 0)
	mcp = new MultiClassSVM(num_train*num_training_passes);
    else if (predictor_type.compare(std::string("multiexp")) == 0)
	mcp = new MultiClassExp(num_train*num_training_passes);
    else if (predictor_type.compare(std::string("multilog")) == 0)
    mcp = new MultiClassLogistic(num_train*num_training_passes);
    else
	mcp = new OneVsAll(num_train*num_training_passes, predictor_type);

    // adjust training data
    std::vector<int> train_label_count(NUM_CLASSES, 0);
    std::vector<double> class_weight(NUM_CLASSES, 0);
    std::vector<int> class_iterations(NUM_CLASSES, 1);
    if(adjust_for_under_represented_classes){
	printf("adjusting for underrepresented classes...\n");
	for (size_t i = 0; i < num_train; ++i)
	    train_label_count[train_labels[i]]++;
	double min_weight = std::numeric_limits<double>::max();
	for (size_t i = 0; i < NUM_CLASSES; ++i){
	    class_weight[i] = double(num_train)/NUM_CLASSES/train_label_count[i];
	    if(class_weight[i] < min_weight)
		min_weight = class_weight[i];
	}
	for (size_t i = 0; i < NUM_CLASSES; ++i){
	    class_iterations[i] = round(class_weight[i] / min_weight);
	    printf("    class %lu accounts for %f of the training data and will be repeated %d times\n",
		   i, double(train_label_count[i])/num_train, class_iterations[i]);
	}
    }
    printf("training with %d passes through the data\n", num_training_passes);
    std::cout << std::string(50,'-') << std::endl;

    // train
    for(int k=0; k<num_training_passes; k++)//run through the training set a few times
	for (size_t i = 0; i < num_train; ++i) 
	    for(int j=0; j<class_iterations[train_labels[i]]; j++)
		mcp->pushData(train_feature_vecs[i], train_labels[i]);

    return mcp;
}

double Validator::testPredictor(std::vector<FeatureVec> test_feature_vecs,
				std::vector<Label> test_labels,
				MultiClassPredictor* mcp, bool print_choice) 
{
    size_t num_test = test_labels.size();

    std::vector<double> test_label_count(NUM_CLASSES, 0);
    std::vector<double> test_label_freq(NUM_CLASSES, 0);
    std::vector<double> per_label_accuracy(NUM_CLASSES, 0);
    std::vector<std::vector<double> > confusion_matrix(NUM_CLASSES, std::vector<double>(NUM_CLASSES, 0));
    double accuracy = 0.0;
    
    // test
    std::vector<Label> predicted_labels = getPredictedLabels(test_feature_vecs, mcp);
    for (size_t i = 0; i < num_test; ++i) {
	Label predicted_label = predicted_labels[i];
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
	std::cout << "Test data performance: \n\n";
	std::cout << "Number of test samples: " << num_test << "\n\n";
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
	std::cout << std::string(100,'-') << std::endl << std::endl;
    }

    return accuracy;
}
