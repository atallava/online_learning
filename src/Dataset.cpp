#include <iostream>
#include <fstream>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

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
    printf("(Dataset summary)\n");
    printf("\tRead in %lu points\n", feature_vecs_.size());
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
    printf("\nShuffling data!\n");
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

void Dataset::whitenData()
{
    printf("Whitening!\n");
    size_t N = feature_vecs_.size();
    Eigen::MatrixXd data(N, feature_vecs_[0].size()-1);
    for (size_t i = 0; i < N; i++)
        data.row(i) = Eigen::VectorXd::Map(&feature_vecs_[i][0],feature_vecs_[i].size()-1);
    printf("\tdata size : %ld %ld\n", data.rows(), data.cols());
    
    auto mean = data.colwise().mean();
    auto centered_data = data - Eigen::MatrixXd::Ones(N,1)*mean;
    auto cov = (centered_data).transpose()*(centered_data)/(N-1);

    // std::cout << "covariance" << std::endl;
    // std::cout << cov << std::endl;

    Eigen::EigenSolver<Eigen::MatrixXd> es(cov);

    std::cout << "eigenvalues : \n" << es.eigenvalues() << std::endl;
    std::cout << "eigenvectors : \n" << es.eigenvectors() << std::endl;

    auto Wd = es.eigenvectors().transpose().real();
    Eigen::MatrixXd D = es.eigenvalues().real();

    std::cout << "D\n" << D << std::endl;

    for (int i = 0; i < D.rows() - 1; i++) {
        // printf("eigenvalue %d inv : %f\n", i, 1/std::sqrt(D(i)));
        Wd.row(i) /= std::sqrt(D(i));
    }

    // printf("centered_data size : %ld %ld\n", centered_data.rows(),
    //     centered_data.cols());
    Eigen::MatrixXd whitened_features = (centered_data*Wd.transpose()).real();

    // printf("size of whitened_features : %d %d\n", whitened_features.rows(),
    //                                               whitened_features.cols());
    std::cout << "sanity\n";
    std::cout << whitened_features.transpose()*whitened_features/(N-1);

    for (size_t i = 0; i < feature_vecs_.size(); i++) {
        for (long int j = 0; j < whitened_features.cols() - 1; j++)
            feature_vecs_[i][j] = whitened_features(i,j);
        feature_vecs_[i].back() = 1;
    }
}
