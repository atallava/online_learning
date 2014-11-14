#include <ol/MultiClassKernelSVM.h>
#include <ol/Dataset.h>
#include <ol/Constants.h>

using namespace ol;
using namespace std;

#define APPROX true

MultiClassKernelSVM::MultiClassKernelSVM(int num_rounds, double lambda) : lambda_(lambda) {
  alpha_.resize(NUM_CLASSES);
#if APPROX
  best_alpha_.resize(NUM_CLASSES);
  num_alpha_used_ = 20;
#endif
  current_iteration_ = 0;

  /*
  static int run=0;
  //double params[5] = {0.001, 0.01, 0.1, 0.5, 1.0};
  double params[3] = {0.0001, 0.001, 0.005};
  //double params[3] = {0.000001, 0.00001, 0.0001};
  lambda_ = params[run];
  run++;
  */
  //lambda_ = 0.000000000001;
  //lambda_ = 0.0;
  //gamma_ = 1000;
  gamma_ = 0.9;


  double G_ = 1;
  learning_rate_ = sqrt(std::log(NUM_FEATURES)/num_rounds)/G_;
}

void MultiClassKernelSVM::pushData(const FeatureVec& features, Label label){
  if(current_iteration_>0){
    Label predicted_label = predict(features);
    updateStreamLogs(label, predicted_label);
  }

  if(current_iteration_ % 10000 == 0){
    printf("data point %d\n",current_iteration_);
    //printf("learning rate %f\n",1.0/(current_iteration_+1));
    /*
    printf("best alphas\n");
    for(unsigned int i=0; i<best_alpha_.size(); i++){
      printf("class %d: ", i);
      for(set<pair<int,double> >::iterator it=best_alpha_[i].begin(); it!=best_alpha_[i].end(); it++)
        printf("(%d, %f) ", it->first, it->second);
      printf("\n");
    }
    */
    //std::cin.get();
  }

  current_iteration_++;
  // set the learning rate adaptively
  //learning_rate_ = static_cast<double>(1.0/current_iteration_);

  //regularization
#if APPROX
  for(unsigned int i=0; i<best_alpha_.size(); i++){
    set<pair<int,double>, AlphaCompare> temp_set;
    for(set<pair<int,double> >::iterator it=best_alpha_[i].begin(); it!=best_alpha_[i].end(); it++)
      temp_set.insert(pair<int,double>(it->first, it->second*(1 - 2 * learning_rate_ * lambda_)));
    best_alpha_[i] = temp_set;
  }
#else
  for(unsigned int i=0; i<alpha_.size(); i++)
    for(unsigned int j=0; j<alpha_[i].size(); j++)
      alpha_[i][j] *= (1 - 2 * learning_rate_ * lambda_);
#endif

  //initialize alphas for new data point
  data_.push_back(features);
#if APPROX
  vector<double> temp_alpha(NUM_CLASSES,0);
  for(unsigned int i=0; i<best_alpha_.size(); i++){
    if(label==i)
      continue;
    double score_correct = kernelFunction(label, features);
    double score_incorrect = kernelFunction(i, features);
    //if not correct by a margin, apply adjustment
    if(score_correct < score_incorrect + 1){
      //move correct class closer to the feature vector
      temp_alpha[label] += learning_rate_;
      //move incorrect class away from the feature vector
      temp_alpha[i] -= learning_rate_;
    }
  }
  //see if the new alphas belong in the set of best alphas
  for(unsigned int i=0; i<best_alpha_.size(); i++){
    if(best_alpha_[i].size() < (unsigned int)num_alpha_used_)
      best_alpha_[i].insert(pair<int,double>(data_.size()-1, temp_alpha[i]));
    else{
      set<pair<int,double> >::iterator smallest_alpha =  best_alpha_[i].begin();
      if(fabs(temp_alpha[i]) > fabs(smallest_alpha->second)){
        best_alpha_[i].erase(smallest_alpha);
        best_alpha_[i].insert(pair<int,double>(data_.size()-1, temp_alpha[i]));
      }
    }
  }
#else
  for(unsigned int i=0; i<alpha_.size(); i++)
    alpha_[i].push_back(0);

  //constraints
  for(unsigned int i=0; i<alpha_.size(); i++){
    if(label==i)
      continue;
    double score_correct = kernelFunction(label, features);
    double score_incorrect = kernelFunction(i, features);
    //if not correct by a margin, apply adjustment
    if(score_correct < score_incorrect + 1){
      //move correct class closer to the feature vector
      alpha_[label].back() += learning_rate_;
      //move incorrect class away from the feature vector
      alpha_[i].back() -= learning_rate_;
    }
  }
#endif

}

Label MultiClassKernelSVM::predict(const FeatureVec& features){
  //printf("predict\n");
  double max_score = std::numeric_limits<double>::min();
  int best_class = -1;
  for(unsigned int i=0; i<alpha_.size(); i++){
    //printf("meh %d\n",i);
    double score = kernelFunction(i, features);
    //printf("class %d scored %f (current best %f)\n",i,score,max_score);
    if(score > max_score){
      max_score = score;
      best_class = i;
      //printf("best class update!\n");
    }
  }
  if(best_class == -1){
    printf("predict returning invalid class!\n");
    exit(6);
  }
  return (Label)best_class;
}

double MultiClassKernelSVM::kernelFunction(int kernel_id, const FeatureVec& features){
  double sum = 0;
#if APPROX
  for(set<pair<int,double> >::iterator it=best_alpha_[kernel_id].begin(); it!=best_alpha_[kernel_id].end(); it++){
    sum += it->second * RBF(features, data_[it->first]);
  }
#else
  for(unsigned int i=0; i<data_.size(); i++){
    if(alpha_[kernel_id][i]==0)
      continue;
    sum += alpha_[kernel_id][i] * RBF(features, data_[i]);
  }
#endif
  if(std::isnan(sum)){
    printf("kernelFunction returning nan!\n");
    exit(5);
  }
  return sum;
}

double MultiClassKernelSVM::RBF(const std::vector<double>& x1, const std::vector<double>& x2){
  double sum = 0;
  for(unsigned int i=0; i<x1.size(); i++){
    if(std::isnan(x1[i]))
      printf("nan x1?\n");
    if(std::isnan(x2[i]))
      printf("nan x2?\n");
    double diff = x1[i]-x2[i];
    if(std::isnan(diff))
      printf("nan diff?\n");
    sum += diff*diff;
  }
  if(std::isnan(exp(-sum/gamma_))){
    printf("underflow! got a nan! sum=%f gamma=%f -sum/gamma=%f (gamma is too small)\n",sum,gamma_,-sum/gamma_);
    exit(4);
  }
  if(exp(-sum/gamma_)<=0){
    printf("underflow! (gamma is too small)\n");
    exit(3);
  }
  return exp(-sum/gamma_);
}

