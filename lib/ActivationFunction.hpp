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
    const activationEnum mSelected;

  public:
    explicit ActivationFunction(activationEnum selected);

    float operator()(float x) const;
    float operator[](float x) const;

  private:
    static float SIGMOID_AC(float x) const;
    static float SIGMOID_CM(float x) const;
    static float TANH_AC(float x) const;
    static float TANH_CM(float x) const;
    static float RELU_AC(float x) const;
    static float RELU_CM(float x) const;
    static float LINEAR_AC(float x) const;
    static float LINEAR_CM(float x) const;
  };
} /* nanoNet */

#endif /* NANONET_ACTIVATIONFUNCTION_HPP */
