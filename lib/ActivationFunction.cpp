#include "ActivationFunction.hpp"

namespace nanoNet {
  ActivationFunction::ActivationFunction(activationEnum selected){
    this->selected = selected;
  }

  void ActivationFunction::setSelected(activationEnum selected) {
    this->selected = selected;
  }

  float ActivationFunction::operator()(float x) const {
    switch (selected) {
      case Sigmoid:
        return sigmoid_activation(x);
      case Tanh:
        return tanh_activation(x);
      case Relu:
        return relu_activation(x);
      case Linear:
        return linear_activation(x);
      default:
        return sigmoid_activation(x);
    }
  }

  float ActivationFunction::operator[](float x) const {
    switch (selected) {
      case Sigmoid:
        return sigmoid_derivative(x);
      case Tanh:
        return tanh_derivative(x);
      case Relu:
        return relu_derivative(x);
      case Linear:
        return linear_derivative(x);
      default:
        return sigmoid_derivative(x);
    }
  }

  float ActivationFunction::sigmoid_activation(float x){
    return 1.0f / ( 1.0f + expf(-x) );
  }
  float ActivationFunction::sigmoid_derivative(float x){
    return x * (x - 1.0f);
  }
  float ActivationFunction::tanh_activation(float x){
    return 2.0f / ( 1.0f + expf(-x) ) - 1.0f;
  }
  float ActivationFunction::tanh_derivative(float x){
    return (1.0f - x)*(1.0f + x);
  }
  float ActivationFunction::relu_activation(float x){
    return std::max(0.0f, x);
  }
  float ActivationFunction::relu_derivative(float x){
    return x > 0.0f ? 1.0f : 0.0f;
  }
  float ActivationFunction::linear_activation(float x){
    return x;
  }
  float ActivationFunction::linear_derivative(float x){
    return 1.0f;
  }
} /* nanoNet */
