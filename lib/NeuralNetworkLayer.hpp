#ifndef NANONET_NEURALNETWORKLAYER_HPP
#define NANONET_NEURALNETWORKLAYER_HPP

#include <ctime>
#include <random>
#include <vector>

#include "ActivationFunction.hpp"

namespace nanoNet {

  using std::vector;
  using std::size_t;

  class NeuralNetworkLayer {
  private:
    ActivationFunction mActivationFunction;
    bool mIsTraining;

    size_t mNodeCount;
    size_t mPrevCount;
    vector<float> mBiases;
    vector<vector<float> > mWeight;

    // TODO: use std::unique_ptr
    /* activation for 'current' train example */
    vector<float>* pValues;

    /* gradients for 'current' train example */
    vector<float>* pValuesG;

    /* gradient sums over all train examples in batch */
    vector<float>* pBiasesGS;
    vector<vector<float> >* pWeightGS;

  public:
    NeuralNetworkLayer();
    NeuralNetworkLayer(size_t nodeCount, size_t prevCount, nanoNet::ActivationFunction::activationEnum activationFunction);
    ~NeuralNetworkLayer();

    vector<float> feedForward(const vector<float>& inputData);

    void gradientFromExample(const vector<float>& exampleData);
    void gradientFromAnother(const NeuralNetworkLayer& next);
    void gradientFromActives(const NeuralNetworkLayer& prev);
    void gradientFromActives(const vector<float>& inputData);

    void substractGradients(float amount);

    void startTraining();
    void stopTraining();
    void applyTraining(float learningRate, int exampleCount);

    size_t getNodeCount() const {return mNodeCount;}
    size_t getPrevCount() const {return mPrevCount;}

    const vector<vector<float> >& getWeight() const {return mWeight;}
  };
} /* nanoNet */

#endif // NANONET_NEURALNETWORKLAYER_HPP
