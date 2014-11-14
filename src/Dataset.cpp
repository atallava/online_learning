#include <iostream>
#include <fstream>

#include <ol/Dataset.h>
#include <stdexcept>

#include <random>

using namespace ol;

Dataset::Dataset(std::string file_name) 
{
    std::ifstream file(file_name);

    // read away junk
    std::string line;
    for (size_t i = 0; i < 3; ++i) {
        std::getline(file,line);
    }

    int tmp;
    double feat;
    while (file) {
        // xyz data
        pcl::PointXYZ point;
        file >> point.x;
        if (!file) 
            break;
        file >> point.y;
        file >> point.z;
        points_.push_back(point);
        
        // id
        file >> tmp;

        // label
        file >> tmp;
        labels_.push_back(mapRawLabelToLabel(tmp));

        // features
        FeatureVec features;
        for (size_t i = 0; i < 10; ++i) {
            file >> feat;
            features.push_back(feat);
        }
        feature_vecs_.push_back(features);
    }
    printf("(Dataset summary)\n");
    printf("\tRead in %d points\n", feature_vecs_.size());
}

Label Dataset::mapRawLabelToLabel(int raw_label) 
{
    Label label;
    switch (raw_label) {
    case 1004:
        // veg
        label = Label::VEG;
        break;
    case 1100:
        // wire
        label = Label::WIRE;
        break;
    case 1103:
        // pole
        label = Label::POLE;
        break;
    case 1200:
        // ground
        label = Label::GROUND;
        break;
    case 1400:
        // facade
        label = Label::FACADE;
        break;
    default:
        throw std::runtime_error("bad raw label!\n");
    }
    return label;
}

void Dataset::shuffleData() 
{
  printf("shuffling data\n");
    std::vector<int> ids;
    for (size_t i = 0; i < labels_.size(); ++i)
	ids.push_back(i);
    
    std::random_shuffle(ids.begin(), ids.end());

    std::vector<Label> tmp_labels(labels_);
    std::vector<pcl::PointXYZ> tmp_points(points_);
    std::vector<FeatureVec> tmp_feature_vecs(feature_vecs_);
    
    for (size_t i = 0; i < ids.size(); ++i) {
	labels_[i] = tmp_labels[ids[i]];
	points_[i] = tmp_points[ids[i]];
	feature_vecs_[i] = tmp_feature_vecs[ids[i]];
    }
}

void Dataset::balanceClasses(){
  int num_classes = 5;
  std::vector<int> label_count(num_classes, 0);
  std::vector<double> class_weight(num_classes, 0);
  std::vector<int> class_duplicates(num_classes, 1);

  printf("adjusting for underrepresented classes...\n");
  for(unsigned int i=0; i<labels_.size(); ++i)
    label_count[labels_[i]]++;
  double min_weight = std::numeric_limits<double>::max();
  for(unsigned int i=0; i<num_classes; ++i){
    class_weight[i] = double(feature_vecs_.size())/num_classes/label_count[i];
    if(class_weight[i] < min_weight)
      min_weight = class_weight[i];
  }
  for(int i=0; i<num_classes; ++i){
    class_duplicates[i] = round(class_weight[i] / min_weight);
    //printf("    class %lu accounts for %f of the training data and will be repeated %d times\n",
        //i, double(train_label_count[i])/num_train, class_iterations[i]);
  }

  std::vector<Label> tmp_labels;
  std::vector<pcl::PointXYZ> tmp_points;
  std::vector<FeatureVec> tmp_feature_vecs;
  for(unsigned int i=0; i<labels_.size(); i++){
    for(int j=0; j<class_duplicates[labels_[i]]; j++){
      tmp_labels.push_back(labels_[i]);
      tmp_points.push_back(points_[i]);
      tmp_feature_vecs.push_back(feature_vecs_[i]);
    }
  }

  labels_ = tmp_labels;
  points_= tmp_points;
  feature_vecs_ = tmp_feature_vecs;
}

void Dataset::addRandomFeatures(){
  for(unsigned int i=0; i<feature_vecs_.size(); i++){
    for(int j=0; j<90; j++){
      feature_vecs_[i].push_back(rand()%100);
    }
  }
}

void Dataset::addNoisyVersionsOfFeatures(){
  std::vector<double> feature_min(feature_vecs_[0].size(),std::numeric_limits<double>::max());
  std::vector<double> feature_max(feature_vecs_[0].size(),std::numeric_limits<double>::min());
  for(unsigned int i=0; i<feature_vecs_.size(); i++){
    for(unsigned int j=0; j<feature_vecs_[i].size(); j++){
      if(feature_vecs_[i][j] < feature_min[j])
        feature_min[j] = feature_vecs_[i][j];
      if(feature_vecs_[i][j] > feature_max[j])
        feature_max[j] = feature_vecs_[i][j];
    }
  }

  std::default_random_engine generator;
  std::vector<std::normal_distribution<double> > gaussians;
  for(int i=0; i<9; i++)
    gaussians.push_back(std::normal_distribution<double>(0.0, 
          0.1*(feature_max[i]-feature_min[i])));

  for(unsigned int i=0; i<feature_vecs_.size(); i++){
    for(int j=0; j<10; j++){
      for(int k=0; k<9; k++){
        double noise = gaussians[k](generator);
        feature_vecs_[i].push_back(feature_vecs_[i][k] + noise);
      }
    }
  }
}



