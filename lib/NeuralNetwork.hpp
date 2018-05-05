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
  struct  DataPoint{
    std::vector<float> input;
    std::vector<float> output;
  };

  class NeuralNetwork{
  private:
    std::size_t inputCount;
    std::size_t outputCount;
    std::size_t layerCount;
    NeuralNetworkLayer outputLayer;
    std::vector<NeuralNetworkLayer> hiddenLayers;
  public:
    NeuralNetwork();
    NeuralNetwork(std::size_t inputCount, std::size_t outputCount);
    ~NeuralNetwork() = default;
    void addLayer(std::size_t nodeCount, ActivationFunction::activationEnum activationFunction);

    std::vector<float> process(const std::vector<float>& inputData);
    void train( const std::vector<DataPoint>& trainData, std::size_t batchSize, std::size_t epochs, float learningRate);

    std::size_t getInputCount(){return inputCount;}
    std::size_t getOutputCount(){return outputCount;}
    std::size_t getLayerCount(){return layerCount;}
  };
} /* nanoNet */



#endif // NANONET_NEURALNETWORK_HPP
