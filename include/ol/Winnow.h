#pragma once
#include <pf/Predictor.h>

namespace ol {
  class Winnow : public Predictor {
  public:
    virtual int predict(std::vector<double> features);
    virtual pushData(std::vector<double> features, int label);
  };
}

