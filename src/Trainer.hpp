#pragma once

#include <cassert>
#include <cstdio>
#include <vector>
#include <random>

#include "NeuralNetwork.hpp"
#include "NeuralNetworkLayer.hpp"

namespace nanoNet
{

struct DataPoint
{
    std::vector<float> input;
    std::vector<float> output;
};

class Trainer
{

    using vector_type = NeuralNetworkLayer::vector_type;
    using matrix_type = NeuralNetworkLayer::matrix_type;

public:
    std::size_t epoch_count = 100;
    std::size_t batch_size = 30;
    float learning_rate = 0.01f;

public:
    using DataSet = std::vector<DataPoint>;

    void train(NeuralNetwork& network, DataSet dataSet)
    {
        std::random_device rd;
        for (size_t i = 0; i < epoch_count; ++i)
        {
            std::shuffle(dataSet.begin(), dataSet.end(), rd);
            train_once(network, { dataSet.begin(), dataSet.begin() + batch_size });
        }
    }

    void train_once(NeuralNetwork& network, const DataSet& dataSet)
    {
        using std::puts;
        using std::size_t;

        std::vector<vector_type> bias_gradient_accum{ network.layerCount() };
        for (size_t i = 0; i < bias_gradient_accum.size(); ++i)
            bias_gradient_accum[i].resize(network.m_layers[i].outputCount(), 0.0);

        std::vector<matrix_type> weight_gradient_accum{ network.layerCount() };
        for (size_t i = 0; i < weight_gradient_accum.size(); ++i)
            weight_gradient_accum[i].resize(network.m_layers[i].outputCount());

        for (size_t i = 0; i < weight_gradient_accum.size(); ++i)
            for (size_t j = 0; j < network.m_layers[i].outputCount(); ++j)
                weight_gradient_accum[i][j].resize(
                    network.m_layers[i].m_weights[j].size(), 0.0);

        std::vector<vector_type> value(network.layerCount());
        std::vector<vector_type> gradient(network.layerCount());
        std::vector<vector_type> bias_gradient(network.layerCount());
        std::vector<matrix_type> weight_gradient(network.layerCount());

        for (auto& dataPoint : dataSet)
        {

            for (size_t i = 0; i < network.layerCount(); ++i)
                value[i]
                    = network.m_layers[i].predict(i == 0 ? dataPoint.input : value[i - 1]);

            gradient.back() = gradientFromExample(value.back(), dataPoint.output);
            for (size_t i = network.layerCount() - 1; i--;)
                gradient[i] = gradientFromNext(network.m_layers[i],
                    network.m_layers[i + 1], value[i + 1], gradient[i + 1]);

            for (size_t i = 0; i < network.layerCount(); ++i)
                bias_gradient[i]
                    = biasGradientFromValues(network.m_layers[i], value[i], gradient[i]);

            weight_gradient[0]
                = weightsGradientFromPrevious(bias_gradient[0], dataPoint.input);
            for (size_t i = 1; i < network.layerCount(); ++i)
                weight_gradient[i]
                    = weightsGradientFromPrevious(bias_gradient[i], value[i - 1]);

            for (size_t i = 0; i < network.layerCount(); ++i)
            {

                for (size_t j = 0; j < bias_gradient[i].size(); ++j)
                    bias_gradient_accum[i][j] += bias_gradient[i][j];

                for (size_t j = 0; j < weight_gradient[i].size(); ++j)
                    for (size_t k = 0; k < weight_gradient[i][j].size(); ++k)
                        weight_gradient_accum[i][j][k] += weight_gradient[i][j][k];
            }
        }

        auto coeff = learning_rate / dataSet.size();
        for (size_t i = 0; i < network.layerCount(); ++i)
        {
            for (size_t j = 0; j < bias_gradient_accum[i].size(); ++j)
                network.m_layers[i].m_biases[j] -= bias_gradient_accum[i][j] * coeff;

            for (size_t j = 0; j < weight_gradient_accum[i].size(); ++j)
                for (size_t k = 0; k < weight_gradient_accum[i][j].size(); ++k)
                    network.m_layers[i].m_weights[j][k]
                        -= weight_gradient_accum[i][j][k] * coeff;
        }
    }

private:
    static vector_type gradientFromExample(
        const vector_type& values, const vector_type& example)
    {
        vector_type gradient(values.size(), 0.0f);

        for (std::size_t i = 0; i < values.size(); ++i)
            gradient[i] = 2 * (values[i] - example[i]);

        return gradient;
    }

    static vector_type gradientFromNext(const NeuralNetworkLayer& layer,
        const NeuralNetworkLayer& next, const vector_type& next_values,
        const vector_type& next_gradient)
    {
        vector_type gradient(layer.outputCount(), 0.0f);

        for (std::size_t i = 0; i < next.outputCount(); ++i)
            for (std::size_t j = 0; j < layer.outputCount(); ++j)
                gradient[j] += next_gradient[i]
                    * next.m_activationFunction[next_values[i]] * next.weights()[i][j];

        return gradient;
    }

    static vector_type biasGradientFromValues(const NeuralNetworkLayer& layer,
        const vector_type& values, const vector_type& gradients)
    {
        vector_type bias_gradients(layer.outputCount(), 0.0f);

        for (std::size_t i = 0; i < layer.outputCount(); ++i)
            bias_gradients[i] = gradients[i] * layer.m_activationFunction[values[i]];

        return bias_gradients;
    }

    static matrix_type weightsGradientFromPrevious(
        const vector_type& bias_gradients, const vector_type& previous_values)
    {
        matrix_type weights_gradients(
            bias_gradients.size(), vector_type(previous_values.size(), 0.0f));

        for (std::size_t i = 0; i < bias_gradients.size(); ++i)
            for (std::size_t j = 0; j < previous_values.size(); ++j)
                weights_gradients[i][j] += bias_gradients[i] * previous_values[j];

        return weights_gradients;
    }
};
}
