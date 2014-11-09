#include <ol/Visualizer.h>

using namespace ol;

void Visualizer::visualize(std::string file_name)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_name, *cloud);

    pcl::visualization::PCLVisualizer viewer("viz");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }     
}
