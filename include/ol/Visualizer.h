#pragma once
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

namespace ol {
    class Visualizer {
    public:
        void visualize(std::string file_name);
    };
}
