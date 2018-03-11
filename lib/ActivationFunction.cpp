#include "ActivationFunction.hpp"

namespace simplemind {
  ActivationFunction::ActivationFunction(activationEnum selected){
    mSelected = selected;
  }

  void ActivationFunction::setSelected(activationEnum selected) {
    mSelected = selected;
  }

  float ActivationFunction::operator()(float x) {
    switch (mSelected) {
      case Sigmoid:
        return SIGMOID_AC(x);
      case Tanh:
        return TANH_AC(x);
      case Relu:
        return RELU_AC(x);
      case Linear:
        return LINEAR_AC(x);
      default:
        return SIGMOID_AC(x);
    }
  }

  float ActivationFunction::operator[](float x) {
    switch (mSelected) {
      case Sigmoid:
        return SIGMOID_CM(x);
      case Tanh:
        return TANH_CM(x);
      case Relu:
        return RELU_CM(x);
      case Linear:
        return LINEAR_CM(x);
      default:
        return SIGMOID_CM(x);
    }
  }

  static float ActivationFunction::SIGMOID_AC(float x){
    return 1.0f / ( 1.0f + expf(-x) );
  }
  static float ActivationFunction::SIGMOID_CM(float x){
    return x * (x - 1.0f);
  }
  static float ActivationFunction::TANH_AC(float x){
    return 2.0f / ( 1.0f + expf(-x) ) - 1.0f;
  }
  static float ActivationFunction::TANH_CM(float x);
  static float ActivationFunction::RELU_AC(float x);
  static float ActivationFunction::RELU_CM(float x);
  static float ActivationFunction::LINEAR_AC(float x);
  static float ActivationFunction::LINEAR_CM(float x);
} /* simplemind */
