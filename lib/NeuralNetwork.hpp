#ifndef SIMPLEMIND_NEURALNETWORK_HPP
#define SIMPLEMIND_NEURALNETWORK_HPP

#include <ctime>
#include <random>
#include <vector>

#include "ActivationFunction.hpp"
#include "NeuralNetworkLayer.hpp"

namespace simplemind{
  class NeuralNetwork{
  private:
    std::size_t mInputCount;
    std::size_t mOutputCount;
    std::size_t mLayerCount;
    NeuralNetworkLayer mOutputLayer;
    std::vector<NeuralNetworkLayer> mHiddenLayers;
  public:
    NeuralNetwork(std::size_t inputCount, std::size_t outputCount);
    void addLayer(std::size_t nodeCount, ActivationFunction::activationEnum activationFunction);

    std::vector<float> process(const std::vector<float>& inputData);
    void train( const std::vector<std::pair<std::vector<float>, std::vector<float> > >& trainData, std::size_t batchSize, std::size_t epochs);

    std::size_t getInputCount(){return mInputCount;}
    std::size_t getOutputCount(){return mOutputCount;}
    std::size_t getLayerCount(){return mLayerCount;}
  };
} /* simplemind */



#endif // SIMPLEMIND_NEURALNETWORK_HPP
