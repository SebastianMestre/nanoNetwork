#ifndef NANONET_NEURALNETWORK_HPP
#define NANONET_NEURALNETWORK_HPP

#include <algorithm>
#include <cstddef>
#include <ctime>
#include <random>
#include <utility>
#include <vector>

#include "ActivationFunction.hpp"
#include "NeuralNetworkLayer.hpp"

namespace nanoNet
{

class NeuralNetwork
{
    friend class Trainer;

private:
    std::vector<NeuralNetworkLayer> m_layers;

public:
    NeuralNetwork();
    ~NeuralNetwork() = default;

    void addLayer(const NeuralNetworkLayer& layer);

    std::vector<float> predict(const std::vector<float>& inputData) const;

    std::size_t layerCount() { return m_layers.size(); }

    std::size_t inputCount() { return layerCount() ? m_layers[0].inputCount() : 0; }

    std::size_t outputCount() { return layerCount() ? m_layers.back().outputCount() : 0; }
};
} /* nanoNet */

#endif // NANONET_NEURALNETWORK_HPP
