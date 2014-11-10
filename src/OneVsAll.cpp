#include <stdexcept>

#include <ol/OneVsAll.h>

using namespace ol;

OneVsAll::OneVsAll(int num_rounds, std::string type)
{
    // initialize the predictors based on the type
    if (type.compare(std::string("logistic")) == 0){
        binary_predictors_.resize(NUM_CLASSES, Logistic(num_rounds));
        binary_labels_ = Logistic::getBinaryLabels();
    } else if (type.compare(std::string("expgraddescent")) == 0) {
        binary_predictors_.resize(NUM_CLASSES, ExpGradDescent(num_rounds));
        binary_labels_ = ExpGradDescent::getBinaryLabels();
    } else {
        throw std::runtime_error("Can't find predictor");
    }
}

