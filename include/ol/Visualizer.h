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
	void visualize(std::vector<pcl::PointXYZ> points_1, std::vector<Label> labels_1,
		       std::vector<pcl::PointXYZ> points_2, std::vector<Label> labels_2);
        void visualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1,
		       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2);
        std::tuple<uint8_t, uint8_t, uint8_t> getLabelColor(Label label);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointsToPCD(std::vector<pcl::PointXYZ> points, 
							   std::vector<Label> labels);
    };
}
