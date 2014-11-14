#include <ol/MultiClassSVM.h>
#include <ol/Dataset.h>
#include <ol/Constants.h>

using namespace ol;

MultiClassSVM::MultiClassSVM(int num_rounds, double lambda) : lambda_(lambda) {
  weights_.resize(NUM_CLASSES);
  for(unsigned int i=0; i<weights_.size(); i++)
    weights_[i].resize(NUM_FEATURES, 1.0/NUM_FEATURES);
  current_iteration_ = 0;

  /*
  static int run=0;
  //double params[5] = {0.001, 0.01, 0.1, 0.5, 1.0};
  double params[3] = {0.0001, 0.001, 0.005};
  //double params[3] = {0.000001, 0.00001, 0.0001};
  lambda_ = params[run];
  run++;
  */
  //lambda_ = 0.0001;

  //double G_ = 1;
  //learning_rate_ = sqrt(std::log(NUM_FEATURES)/num_rounds)/G_;
}

void MultiClassSVM::pushData(const FeatureVec& features, Label label){
  // update stream log
  Label predicted_label = predict(features);
  updateStreamLogs(label, predicted_label);

  current_iteration_++;
  // set the learning rate adaptively
  learning_rate_ = static_cast<double>(1.0/sqrt(current_iteration_));

  //regularization
  for(unsigned int i=0; i<weights_.size(); i++)
    for(unsigned int j=0; j<weights_[i].size(); j++)
      weights_[i][j] -= learning_rate_ * lambda_ * weights_[i][j];

  //constraints
  for(unsigned int i=0; i<weights_.size(); i++){
    if(label==i)
      continue;
    double wf_correct = std::inner_product(weights_[label].begin(), weights_[label].end(),
        features.begin(), 0.0);
    double wf_incorrect = std::inner_product(weights_[i].begin(), weights_[i].end(),
        features.begin(), 0.0);
    //if not correct by a margin, apply adjustment
    if(wf_correct < wf_incorrect + 1){
      for(unsigned int j=0; j<weights_[i].size(); j++){
        //move correct class closer to the feature vector
        weights_[label][j] -= learning_rate_ * -features[j]; 
        //move incorrect class away from the feature vector
        weights_[i][j]     -= learning_rate_ * features[j];
      }
    }
  }
}

Label MultiClassSVM::predict(const FeatureVec& features){
  double max_wf = std::numeric_limits<double>::min();
  int best_class = -1;
  for(unsigned int i=0; i<weights_.size(); i++){
    double wf = std::inner_product(weights_[i].begin(), weights_[i].end(),
                                   features.begin(), 0.0);
    if(wf > max_wf){
      max_wf = wf;
      best_class = i;
    }
  }
  return (Label)best_class;
}

