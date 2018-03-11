#ifndef NANONET_ACTIVATIONFUNCTION_HPP
#define NANONET_ACTIVATIONFUNCTION_HPP

#include <cmath>

namespace nanoNet {
  class ActivationFunction{

  public:
    enum activationEnum{Sigmoid = 0, Tanh, Relu, Linear};
  private:
    activationEnum mSelected;

  public:
    ActivationFunction(activationEnum selected);
    void setSelected(activationEnum selected);

    float operator()(float x) const;
    float operator[](float x) const;

  private:
    static float SIGMOID_AC(float x);
    static float SIGMOID_CM(float x);
    static float TANH_AC(float x);
    static float TANH_CM(float x);
    static float RELU_AC(float x);
    static float RELU_CM(float x);
    static float LINEAR_AC(float x);
    static float LINEAR_CM(float x);
  };
} /* nanoNet */

#endif /* NANONET_ACTIVATIONFUNCTION_HPP */
