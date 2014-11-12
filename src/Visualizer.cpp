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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = pointsToPCD(points, labels);
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
        viewer.spinOnce(100);
    }     
}

void Visualizer::visualize(std::vector<pcl::PointXYZ> points_1, std::vector<Label> labels_1,
			   std::vector<pcl::PointXYZ> points_2, std::vector<Label> labels_2)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1 = pointsToPCD(points_1, labels_1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2 = pointsToPCD(points_2, labels_2);
    visualize(cloud_1, cloud_2);
}

void Visualizer::visualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1,
			   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2)
{
  pcl::visualization::PCLVisualizer viewer("viz");
  viewer.initCameraParameters();

  int v1(0);
  viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
  viewer.addText("Cloud 1", 10, 10, "v1 text", v1);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_1(cloud_1);
  viewer.addPointCloud<pcl::PointXYZRGB> (cloud_1, rgb_1, "cloud_v1", v1);
  viewer.addPointCloud(cloud_1, rgb_1, "cloud_1", v1);
  
  int v2(0);
  viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
  viewer.addText("Cloud 2", 10, 10, "v2 text", v2);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_2(cloud_2);
  viewer.addPointCloud<pcl::PointXYZRGB> (cloud_2, rgb_2, "cloud_v2", v2);
  viewer.addPointCloud(cloud_2, rgb_2, "cloud_2", v2);

  viewer.addCoordinateSystem(1.0);
  
  while (!viewer.wasStopped()) {
    viewer.spinOnce(100);
  }
}

std::tuple<uint8_t, uint8_t, uint8_t> Visualizer::getLabelColor(Label label)
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
    // uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    // return rgb;
    return std::make_tuple(r, g, b);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Visualizer::pointsToPCD(std::vector<pcl::PointXYZ> points, 
							       std::vector<Label> labels)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PointXYZRGB pointRGB;
    for (size_t i = 0; i < points.size(); ++i) {
        pointRGB.x = points[i].x;
        pointRGB.y = points[i].y;            
        pointRGB.z = points[i].z;
        auto rgb = getLabelColor(labels[i]);
        pointRGB.r = std::get<0>(rgb);
        pointRGB.g = std::get<1>(rgb);
        pointRGB.b = std::get<2>(rgb);
        cloud->push_back(pointRGB);
    }
    
    return cloud;
}
