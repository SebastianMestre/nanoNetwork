#ifndef NANONET_NEURALNETWORK_HPP
#define NANONET_NEURALNETWORK_HPP

#include <algorithm>
#include <ctime>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "ActivationFunction.hpp"
#include "NeuralNetworkLayer.hpp"

namespace nanoNet{

  class NeuralNetwork{
  private:
    std::vector<NeuralNetworkLayer> hiddenLayers;
  public:
    NeuralNetwork();
    ~NeuralNetwork() = default;


    void addLayer (const NeuralNetworkLayer& layer);

    std::vector<float> predict(const std::vector<float>& inputData) const;


    std::size_t layerCount () {
        return hiddenLayers.size();
    }

    std::size_t inputCount () {
        return layerCount()
        ? hiddenLayers[0].inputCount()
        : 0;
    }

    std::size_t outputCount () {
        return layerCount()
        ? hiddenLayers.back().outputCount()
        : 0;
    }
  };
} /* nanoNet */



#endif // NANONET_NEURALNETWORK_HPP
