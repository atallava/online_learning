#include <pcl/common/common_headers.h>
#include <ol/Visualizer.h>

using namespace ol;

void Visualizer::visualize(std::string file_name)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(file_name, *cloud);
    visualize(cloud);
}

void Visualizer::visualize(std::vector<pcl::PointXYZ> points, std::vector<Label> labels)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PointXYZRGB pointRGB;
    for (size_t i = 0; i < points.size(); ++i) {
        pointRGB.x = points[i].x;
        pointRGB.y = points[i].y;            
        pointRGB.z = points[i].z;
        pointRGB.rgb = getLabelColour(labels[i]);
        cloud->push_back(pointRGB);
    }

    visualize(cloud);
}

void Visualizer::visualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::visualization::PCLVisualizer viewer("viz");
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();
    viewer.setCameraPosition(180,180,0,0,0,1);

    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }     
}

uint32_t Visualizer::getLabelColour(Label label)
{
    uint8_t r, g, b;
    switch (label) {
    case 0:
        r = 0; g = 255; b = 0;
        break;
    case 1:
        r = 128; g = 128; b = 128;
        break;
    case 2:
        r = 0; g = 0; b = 205;
        break;
    case 3:
        r = 128; g = 0; b = 0;
        break;
    case 4:
        r = 255; g = 255; b = 255;
        break;
    default:
        throw std::invalid_argument("Invalid label");
    }
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    return rgb;
}
