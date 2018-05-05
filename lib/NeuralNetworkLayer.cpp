#include <vector>
#include "NeuralNetworkLayer.hpp"

namespace nanoNet {
  NeuralNetworkLayer::NeuralNetworkLayer(size_t nodeCount, size_t prevCount, ActivationFunction::activationEnum activationFunction)
  : mActivationFunction(activationFunction) {

    // std::mt19937 rng(std::time(NULL));
    // std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    mNodeCount = nodeCount;
    mPrevCount = prevCount;
    mIsTraining = false;

    mBiases = vector<float>(nodeCount, 0.0f);
    mWeight = vector<vector<float> >(nodeCount, vector<float>(prevCount, 0.0f));

    for(int i = 0; i < nodeCount; i++){
      // mBiases[i] = uniform(rng);
      mBiases[i] = 0.0f;
      for(int j = 0; j < prevCount; j++){
        // mWeight[i][j] = uniform(rng);
        mWeight[i][j] = 0.0f;
      }
    }
  }

  NeuralNetworkLayer::~NeuralNetworkLayer(){
    // TODO: complete this
    if(mIsTraining){
      stopTraining();
    }
  }

  vector<float> NeuralNetworkLayer::feedForward(const vector<float>& inputData){

    vector<float> result(mNodeCount);

    for (int i = 0; i < mNodeCount; i++) {
      result[i] = mBiases[i];
      for (int j = 0; j < mPrevCount; j++) {
        result[i] += mWeight[i][j] * inputData[j];
      }
      result[i] = mActivationFunction(result[i]);
    }

    if(mIsTraining){
      (*pValues) = result;
    }

    return result;
  }

  void NeuralNetworkLayer::gradientFromExample(const vector<float>& exampleData) {
    if(!mIsTraining)
      return;

    for (int i = 0; i < mNodeCount; i++) {
      (*pValuesG)[i] = 2 * ((*pValues)[i] - exampleData[i]);
    }
  }

  void NeuralNetworkLayer::gradientFromAnother(const NeuralNetworkLayer& next) {
    if(!mIsTraining)
      return;

    auto nextWeight = next.getWeight();
    for (size_t i = 0; i < mNodeCount; i++) {
      (*pValuesG)[i] = 0.0f;
      for (int j = 0; j < next.mNodeCount; j++) {
        (*pValuesG)[i] += (*next.pValuesG)[j] * next.mActivationFunction[ (*next.pValues)[j] ] * nextWeight[j][i];
      }

    }
  }

  void NeuralNetworkLayer::gradientFromActives(const NeuralNetworkLayer& prev) {
    gradientFromActives(*prev.pValues);
  }

  void NeuralNetworkLayer::gradientFromActives(const vector<float>& inputData) {
    if(!mIsTraining)
      return;

    // TODO: fix names
    for(int i = 0; i < mNodeCount; i++){
      float aaa = (*pValuesG)[i] * mActivationFunction[ (*pValues)[i] ];

      (*pBiasesGS)[i] += aaa;

      for (int j = 0; j < mPrevCount; j++) {

        (*pWeightGS)[i][j] += aaa * inputData[j];

      }
    }
  }

  void NeuralNetworkLayer::substractGradients(float amount){
    if(!mIsTraining)
      return;

    for (int i = 0; i < mNodeCount; i++) {
      mBiases[i] -= (*pBiasesGS)[i] * amount;
      for (int j = 0; j < mPrevCount; j++) {
        mWeight[i][j] -= (*pWeightGS)[i][j] * amount;
      }
    }
  }

  void NeuralNetworkLayer::startTraining() {
    if(!mIsTraining)
      return;

    pValues = new vector<float>(mNodeCount, 0.0f);

    pValuesG = new vector<float>(mNodeCount, 0.0f);

    pBiasesGS = new vector<float>(mNodeCount, 0.0f);
    pWeightGS = new vector<vector<float> >(mNodeCount, vector<float>(mPrevCount, 0.0f));

    mIsTraining = true;
  }

  void NeuralNetworkLayer::stopTraining() {
    if(!mIsTraining)
      return;

    delete pValues;

    delete pValuesG;

    delete pBiasesGS;
    delete pWeightGS;

    mIsTraining = false;
  }

  void NeuralNetworkLayer::applyTraining(float learningRate, int exampleCount){
    substractGradients(learningRate * exampleCount);

    std::fill(pBiasesGS->begin(), pBiasesGS->end(), 0.0f);
    for(auto& v : *pWeightGS){
      std::fill(v.begin(), v.end(), 0.0f);
    }
  }


} /* nanoNet */
