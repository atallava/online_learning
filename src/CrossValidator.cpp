#include <numeric>

#include <ol/CrossValidator.h>

using namespace ol;

CrossValidator::CrossValidator(CrossValidatorParams params)
    :   params_(params),
        validator_(new Validator()),
        train_dataset_(params.train_file_name)
{
    // modify the dataset as necessary
    train_dataset_.balanceClasses();
    train_dataset_.shuffleData();
    printf("[CrossValidator] Received dataset of size : %lu\n",
        train_dataset_.size());
}



ValidatorParams CrossValidator::getBestParameter()
{
    // create validator params here from crossValidatorParams and get the
    // average accuracy
    auto validator_params = params_.getValidatorParams();
    // implement only regularization for now.
    double best_accuracy = 0.0;
    ValidatorParams best_params;
    // validator_params.lambda = params_.regularization.lower_limit/10;
    validator_params.lambda = params_.regularization.lower_limit;
    for (int i = 0; validator_params.lambda < params_.regularization.upper_limit; i++) {
        // uniform search for now on a logarithmic scale
        // validator_params.lambda = params_.regularization.lower_limit +
        //     i*(params_.regularization.upper_limit -
        //     params_.regularization.lower_limit)/params_.regularization.num_points;
        double accuracy = averageAccuracyForParameters(validator_params);
        if (best_accuracy < accuracy) {
            best_accuracy = accuracy;
            best_params = validator_params;
        }
        printf("\nevaluated lambda : %f at %f\n\n\n", validator_params.lambda,
            accuracy);
        validator_params.lambda *= 1000;
    }
    return best_params;
}

double CrossValidator::averageAccuracyForParameters(ValidatorParams
    validator_params)
{
    std::vector<double> fold_accuracies(params_.num_folds, 0.0);
    // Pass in the training and test points and
    // get average accuracy.
    for (int i = 0; i < params_.num_folds; i++) {
        // get the test chunk indices
        auto testset = getTestFoldIndices(i);
        
        // printf("\tfold number : %d : test indices are : %d\t%d\n", i,
        //     testset.first, testset.second);
        // todo : implement this format
        validator_->trainPredictor(train_dataset_, testset,
            params_.predictor_type, validator_params);
        fold_accuracies[i] = validator_->testPredictor(train_dataset_, testset);
        printf("\tfold number : %d : accuracy : %f\n", fold_accuracies[i]);
    }
    double average_accuracy = 0.0;
    std::for_each(fold_accuracies.begin(), fold_accuracies.end(),
        [&average_accuracy](double a) {
            average_accuracy += a;
        });
    average_accuracy/=fold_accuracies.size();
    return average_accuracy;
    // return std::accumulate(fold_accuracies.begin(), fold_accuracies.end(),
    //     0.0)/fold_accuracies.size();
}

// fold_id must be 0 to num_folds - 1
std::pair<int,int> CrossValidator::getTestFoldIndices(int fold_id)
{
    // data is already shuffled. We only need to return the lower and upper
    // indices for the testset
    // let the errors accumulate at the end. If we have, for example, 1234
    // points and we want to chunk it into 10 folds, we will have 123 points
    // in each (take the floor) and so, in the first 9 folds, we'll have 123*9 =
    // 1107 points and in the last one, we'll have 127 points.

    // format used is always [begin, end)
    int begin, end;
    int chunk_size = static_cast<int>(train_dataset_.size()/params_.num_folds);
    begin = fold_id*chunk_size;
    if (fold_id == params_.num_folds - 1)
        end = train_dataset_.size();
    else
        end = begin + chunk_size;
    return std::make_pair(begin, end);
}

ValidatorParams CrossValidatorParams::getValidatorParams()
{
    ValidatorParams validator_params;
    validator_params.num_training_passes = num_training_passes;
    return validator_params;
}