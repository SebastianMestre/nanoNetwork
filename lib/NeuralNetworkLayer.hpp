#ifndef NANONET_NEURALNETWORKLAYER_HPP
#define NANONET_NEURALNETWORKLAYER_HPP

#include <ctime>
#include <random>
#include <vector>

#include "ActivationFunction.hpp"

namespace nanoNet {

  class NeuralNetworkLayer {
  private:
    ActivationFunction mActivationFunction;
    bool mIsTraining;

    std::size_t mNodeCount;
    std::size_t mPrevCount;
    std::vector<float> mBiases;
    std::vector<std::vector<float> > mWeight;

    // TODO: use std::unique_ptr
    /* activation for 'current' train example */
    std::vector<float>* pValues;

    /* gradients for 'current' train example */
    std::vector<float>* pValuesG;

    /* gradient sums over all train examples in batch */
    std::vector<float>* pBiasesGS;
    std::vector<std::vector<float> >* pWeightGS;

  public:
    NeuralNetworkLayer();
    NeuralNetworkLayer(std::size_t nodeCount, std::size_t prevCount, nanoNet::ActivationFunction::activationEnum activationFunction);
    ~NeuralNetworkLayer();

    std::vector<float> feedForward(const std::vector<float>& inputData);

    void gradientFromExample(const std::vector<float>& exampleData);
    void gradientFromAnother(const NeuralNetworkLayer& next);
    void gradientFromActives(const NeuralNetworkLayer& prev);
    void gradientFromActives(const std::vector<float>& inputData);

    void substractGradients(float amount);

    void startTraining();
    void stopTraining();
    void applyTraining(float learningRate, int exampleCount);

    std::size_t getNodeCount() const {return mNodeCount;}
    std::size_t getPrevCount() const {return mPrevCount;}

    const std::vector<std::vector<float> >& getWeight() const {return mWeight;}
  };
} /* nanoNet */

#endif // NANONET_NEURALNETWORKLAYER_HPP
