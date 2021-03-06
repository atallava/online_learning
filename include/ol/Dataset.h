#pragma once
#include <string>
#include <vector>
#include <ol/Constants.h>
#include <pcl/point_types.h>

namespace ol {
    typedef std::vector<double> FeatureVec;

    class Dataset {
    public:
        Dataset(std::string file_name);
        Label mapRawLabelToLabel(int raw_label);
        std::vector<Label> labels() const { return labels_; }
        std::vector<pcl::PointXYZ> points() const { return points_; }
        std::vector<FeatureVec> feature_vecs() const { return feature_vecs_; }
        void shuffleData();
        void balanceClasses();
        void addRandomFeatures();
        void addNoisyVersionsOfFeatures();
        size_t size() const  { return feature_vecs_.size(); }  // should be the same as labels_.size() too
    private:
        std::vector<Label> labels_;
        std::vector<pcl::PointXYZ> points_;
        std::vector<FeatureVec> feature_vecs_;
    };
}
