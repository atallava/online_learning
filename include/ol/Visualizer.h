#pragma once
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <ol/Dataset.h>

namespace ol {
    class Visualizer {
    public:
        void visualize(std::string file_name);
        void visualize(std::vector<pcl::PointXYZ> points, std::vector<Label> labels);
        void visualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
        uint32_t getLabelColour(Label label);
    };
}
