#ifndef NANONET_ACTIVATIONFUNCTION_HPP
#define NANONET_ACTIVATIONFUNCTION_HPP

#include <cmath>
#include <utility>
#include <algorithm>

namespace nanoNet {
  class ActivationFunction{

  public:
    enum activationEnum{Sigmoid = 0, Tanh, Relu, Linear};
  private:
    activationEnum selected;

  public:
    ActivationFunction(activationEnum selected);
    void setSelected(activationEnum selected);

    float operator()(float x) const;
    float operator[](float x) const;

  private:
    static float sigmoid_activation(float x);
    static float sigmoid_derivative(float x);
    static float tanh_activation(float x);
    static float tanh_derivative(float x);
    static float relu_activation(float x);
    static float relu_derivative(float x);
    static float linear_activation(float x);
    static float linear_derivative(float x);
  };
} /* nanoNet */

#endif /* NANONET_ACTIVATIONFUNCTION_HPP */
