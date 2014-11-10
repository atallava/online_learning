#include <ol/Validator.h>
#include <ol/Dataset.h>
#include <ol/OneVsAll.h>
#include <ol/Constants.h>

using namespace ol;

double Validator::validateOnDataset(std::string file_name, std::string predictor_type) 
{
    OneVsAll ova(num_train_, predictor_type);
    Dataset dset(file_name);
    std::vector<FeatureVec> feature_vecs = dset.feature_vecs();
    std::vector<Label> labels = dset.labels();
    for (size_t i = 0; i < num_train_; ++i) 
        ova.pushData(feature_vecs[i], labels[i]);

    double accuracy = 0.0;
    Label predicted_label;
    for (size_t i = 0; i < num_test_ && i < feature_vecs.size(); ++i) {
        predicted_label = ova.predict(feature_vecs[num_train_+i]);
        // printf("actual : %d, predicted_label : %d\n", labels[num_train_+i],
        //                                                  predicted_label);
        if (predicted_label == labels[num_train_+i])
            accuracy = accuracy+1;
    }
    accuracy = accuracy/num_test_;
    return accuracy;
}
