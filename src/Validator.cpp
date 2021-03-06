#include <iomanip>
#include <ctime>
#include <ol/Validator.h>
#include <ol/Dataset.h>
#include <ol/OneVsAll.h>
#include <ol/MultiClassSVM.h>
#include <ol/MultiClassKernelSVM.h>
#include <ol/MultiClassExp.h>
#include <ol/MultiClassLogistic.h>
#include <ol/Constants.h>

using namespace ol;

double Validator::validate(std::string file_name, std::string predictor_type, double predictor_param,
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
		    predictor_type, predictor_param, print_choice,
		    adjust_for_under_represented_classes,
		    num_training_passes);
}

double Validator::validate(std::string train_file_name, std::string test_file_name, 
			   std::string predictor_type, double predictor_param,
			   bool print_choice, bool viz_choice,
			   bool adjust_for_under_represented_classes,
			   int num_training_passes)
{
    Dataset train_dset(train_file_name);
    train_dset.balanceClasses();
    train_dset.shuffleData();
    Dataset test_dset(test_file_name);

    //train_dset.addRandomFeatures();
    //test_dset.addRandomFeatures();
    //train_dset.addNoisyVersionsOfFeatures();
    //test_dset.addNoisyVersionsOfFeatures();

    // form train and test
    std::vector<FeatureVec> train_feature_vecs = train_dset.feature_vecs();
    std::vector<Label> train_labels = train_dset.labels();
    std::vector<FeatureVec> test_feature_vecs = test_dset.feature_vecs();
    std::vector<Label> test_labels = test_dset.labels();
    
    std::clock_t begin = std::clock();    
    MultiClassPredictor* mcp = trainPredictor(train_feature_vecs, train_labels, predictor_type, predictor_param,
					      adjust_for_under_represented_classes, num_training_passes);
    std::clock_t end = std::clock();
    double elapsed_time = double(end-begin)/CLOCKS_PER_SEC;

    if (print_choice) {
	std::cout << "Predictor: " << predictor_type << "\n\n";
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
      printf("visualize!\n");
	std::vector<Label> predicted_labels = getPredictedLabels(test_feature_vecs, mcp);
	Visualizer vizer;
	// visualize ground truth and predictions side-by-side
	// vizer.visualize(test_dset.points(), test_labels,
	// 		test_dset.points(), predicted_labels);

	std::string file_name;
	file_name = "pcl_viz/" + predictor_type + "_test.png";
	vizer.setFileLocation(file_name);
	// visualize only predictions; write to disk
	vizer.visualize(test_dset.points(), predicted_labels);
    }

    delete mcp;
    return accuracy;
}


double Validator::validate(std::vector<FeatureVec> train_feature_vecs, std::vector<Label> train_labels, 
			   std::vector<FeatureVec> test_feature_vecs, std::vector<Label> test_labels, 
			   std::string predictor_type, double predictor_param,
			   bool print_choice, bool adjust_for_under_represented_classes,
			   int num_training_passes)
{
    MultiClassPredictor* mcp = trainPredictor(train_feature_vecs, train_labels, predictor_type, predictor_param,
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
					       std::string predictor_type, double predictor_param,
					       bool adjust_for_under_represented_classes, 
					       int num_training_passes)
{
    size_t num_train = train_labels.size();

    //normalize features
    /*
    std::vector<double> feature_min(train_feature_vecs[0].size(),std::numeric_limits<double>::max());
    std::vector<double> feature_max(train_feature_vecs[0].size(),std::numeric_limits<double>::min());
    for(unsigned int i=0; i<train_feature_vecs.size(); i++){
      for(unsigned int j=0; j<train_feature_vecs[i].size(); j++){
        if(train_feature_vecs[i][j] < feature_min[j])
          feature_min[j] = train_feature_vecs[i][j];
        if(train_feature_vecs[i][j] > feature_max[j])
          feature_max[j] = train_feature_vecs[i][j];
      }
    }
    for(unsigned int i=0; i<feature_max.size(); i++){
      if(feature_max[i]-feature_min[i] == 0.0){
        feature_max[i] = 1;
        feature_min[i] = 0;
      }
    }
    for(unsigned int i=0; i<feature_max.size(); i++)
      printf("feature %d: min=%f max=%f range=%f\n",i,feature_min[i],feature_max[i],feature_max[i]-feature_min[i]);
    for(unsigned int i=0; i<train_feature_vecs.size(); i++)
      for(unsigned int j=0; j<train_feature_vecs[i].size(); j++)
        train_feature_vecs[i][j] = (train_feature_vecs[i][j]-feature_min[j])/(feature_max[j]-feature_min[j]);
    for(unsigned int i=0; i<test_feature_vecs.size(); i++)
      for(unsigned int j=0; j<test_feature_vecs[i].size(); j++)
        test_feature_vecs[i][j] = (test_feature_vecs[i][j]-feature_min[j])/(feature_max[j]-feature_min[j]);
        */

    // create predictor
    MultiClassPredictor* mcp;
    if (predictor_type.compare(std::string("svm")) == 0)
      mcp = new MultiClassSVM(num_train*num_training_passes, predictor_param);
    else if (predictor_type.compare(std::string("kernel_svm")) == 0)
      mcp = new MultiClassKernelSVM(num_train*num_training_passes, predictor_param);
    else if (predictor_type.compare(std::string("multiexp")) == 0)
      mcp = new MultiClassExp(num_train*num_training_passes, predictor_param);
    else if (predictor_type.compare(std::string("multilog")) == 0)
      mcp = new MultiClassLogistic(num_train*num_training_passes, predictor_param);
    else
      mcp = new OneVsAll(num_train*num_training_passes, predictor_type, predictor_param);

    printf("training with %d passes through the data\n", num_training_passes);
    std::cout << std::string(50,'-') << std::endl;

    // train
    for(int k=0; k<num_training_passes; k++)//run through the training set a few times
      for (size_t i = 0; i < num_train; ++i) 
        mcp->pushData(train_feature_vecs[i], train_labels[i]);

    return mcp;
}

double Validator::testPredictor(std::vector<FeatureVec> test_feature_vecs,
				std::vector<Label> test_labels,
				MultiClassPredictor* mcp, bool print_choice) 
{
    std::clock_t begin = std::clock();    
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
    std::clock_t end = std::clock();
    double elapsed_time = double(end-begin)/CLOCKS_PER_SEC;

    // pretty printing
    if (print_choice) {
	std::cout << "Test data performance: \n\n";
	printf("Test time (CPU): %0.2fs\n\n", elapsed_time);
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

void Validator::trainPredictor(const Dataset& dataset, std::pair<int,int>
    testset,std::string predictor_type, ValidatorParams params)
{
    std::vector<FeatureVec> features = dataset.feature_vecs();
    std::vector<Label> labels = dataset.labels();

    // transform the params to the multi class params
    auto predictor_params = params.getPredictorParams();
    predictor_params.num_rounds = dataset.size() -
                                  (testset.second - testset.first);

    // create the right instances
    if (predictor_type.compare(std::string("svm")) == 0)
        predictor_.reset(new MultiClassSVM(predictor_params));
    else if (predictor_type.compare(std::string("kernel_svm")) == 0)
        predictor_.reset(new MultiClassKernelSVM(predictor_params));
    else if (predictor_type.compare(std::string("multiexp")) == 0)
        predictor_.reset(new MultiClassExp(predictor_params));
    else if (predictor_type.compare(std::string("multilog")) == 0)
        predictor_.reset(new MultiClassLogistic(predictor_params));
    else
        predictor_.reset(new OneVsAll(predictor_type, predictor_params));
    
    printf("training with %d passes through the data\n",
                            params.num_training_passes);
    std::cout << std::string(50,'-') << std::endl;

    // train!
    for(int k=0; k<params.num_training_passes; k++) {
      for (size_t i = 0; i < dataset.size(); ++i) {
        // skip the testset
        if (i >= testset.first && i < testset.second) {
            ;
        } else { 
            predictor_->pushData(features[i], labels[i]);
        }
      }
    }
}

double Validator::testPredictor(const Dataset& dataset, std::pair<int,int>
    testset)
{
    size_t num_test = testset.second - testset.first;
    // printf("num test %d: %d %d\n", num_test, testset.first, testset.second);

    // printf("size of featurevecs : %lu\n", dataset.feature_vecs().size());
    std::vector<FeatureVec> test_feature_vecs;
    auto all_feature_vecs = dataset.feature_vecs();
    for (int i = testset.first; i < testset.second; i++)
        test_feature_vecs.push_back(all_feature_vecs[i]);

    auto all_labels = dataset.labels();
    std::vector<Label> test_labels;
    for (int i = testset.first; i < testset.second; i++)
        test_labels.push_back(all_labels[i]);
    

    std::vector<double> test_label_count(NUM_CLASSES, 0);
    std::vector<double> test_label_freq(NUM_CLASSES, 0);
    std::vector<double> per_label_accuracy(NUM_CLASSES, 0);
    std::vector<std::vector<double> > confusion_matrix(NUM_CLASSES, std::vector<double>(NUM_CLASSES, 0));
    double accuracy = 0.0;

    // test
    for (size_t i = 0; i < num_test; ++i) {
        Label predicted_label = predictor_->predict(test_feature_vecs[i]);
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

    double overall_per_label_acc = 0.0;
    overall_per_label_acc = std::accumulate(per_label_accuracy.begin(), per_label_accuracy.end(), 0.0);
    overall_per_label_acc /= per_label_accuracy.size();
    return overall_per_label_acc;
    // printf("\t\t\taccuracy on testset : %f\n", accuracy);
    // return accuracy;
}

MultiClassPredictorParams ValidatorParams::getPredictorParams()
{
    MultiClassPredictorParams params;
    params.lambda = lambda;
    return params;
}
