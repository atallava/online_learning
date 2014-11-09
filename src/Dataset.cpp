#include <iostream>
#include <fstream>

#include <ol/Dataset.h>

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
}

Label Dataset::mapRawLabelToLabel(int raw_label) 
{
    Label label;
    switch (raw_label) {
    case 1004:
        // veg
        label = 0;
        break;
    case 1100:
        // wire
        label = 1;
        break;
    case 1103:
        // pole
        label = 2;
        break;
    case 1200:
        // ground
        label = 3;
        break;
    case 1400:
        // facade
        label = 4;
        break;
    default:
        printf("here!\n");
        label = -1;
    }
    return label;
}
