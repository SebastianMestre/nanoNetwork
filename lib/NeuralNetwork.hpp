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

  using std::vector;

  struct  DataPoint{
    vector<float> input;
    vector<float> output;
  };

  class NeuralNetwork{
  private:
    size_t mInputCount;
    size_t mOutputCount;
    size_t mLayerCount;
    NeuralNetworkLayer mOutputLayer;
    vector<NeuralNetworkLayer> mHiddenLayers;
  public:
    NeuralNetwork();
    NeuralNetwork(size_t inputCount, size_t outputCount);
    ~NeuralNetwork() = default;
    void addLayer(size_t nodeCount, ActivationFunction::activationEnum activationFunction);

    vector<float> process(const vector<float>& inputData);
    void train( const vector<DataPoint>& trainData, size_t batchSize, size_t epochs, float learningRate);

    size_t getInputCount(){return mInputCount;}
    size_t getOutputCount(){return mOutputCount;}
    size_t getLayerCount(){return mLayerCount;}
  };
} /* nanoNet */



#endif // NANONET_NEURALNETWORK_HPP
