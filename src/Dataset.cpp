#include <iostream>
#include <fstream>

#include <ol/Dataset.h>
#include <stdexcept>

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
