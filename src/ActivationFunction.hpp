#ifndef NANONET_ACTIVATIONFUNCTION_HPP
#define NANONET_ACTIVATIONFUNCTION_HPP

#include <algorithm>
#include <cmath>
#include <utility>

namespace nanoNet
{

class ActivationFunction
{

public:
    enum Which
    {
        Sigmoid = 0,
        Tanh,
        Relu,
        Linear
    };

private:
    Which m_option;

public:
    ActivationFunction(const Which& option);
    void setOption(const Which& option);

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
