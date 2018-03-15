#include "ActivationFunction.hpp"

namespace nanoNet {
  ActivationFunction::ActivationFunction(activationEnum selected) : mSelected(selected) {

  }

  float ActivationFunction::operator()(float x) const {
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

  float ActivationFunction::operator[](float x) const {
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

  float ActivationFunction::SIGMOID_AC(float x) const {
    return 1.0f / ( 1.0f + expf(-x) );
  }
  float ActivationFunction::SIGMOID_CM(float x) const {
    return x * (x - 1.0f);
  }
  float ActivationFunction::TANH_AC(float x) const {
    return 2.0f / ( 1.0f + expf(-x) ) - 1.0f;
  }
  float ActivationFunction::TANH_CM(float x) const {
    return (1.0f - x)*(1.0f + x);
  }
  float ActivationFunction::RELU_AC(float x) const {
    return std::max(0.0f, x);
  }
  float ActivationFunction::RELU_CM(float x) const {
    return x > 0.0f ? 1.0f : 0.0f;
  }
  float ActivationFunction::LINEAR_AC(float x) const {
    return x;
  }
  float ActivationFunction::LINEAR_CM(float x) const {
    return 1.0f;
  }
} /* nanoNet */
