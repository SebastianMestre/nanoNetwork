#include "NeuralNetworkLayer.hpp"

namespace simplemind {

  NeuralNetworkLayer::NeuralNetworkLayer(std::size_t nodeCount, std::size_t prevCount, ActivationFunction::activationEnum activationFunction){
    std::mt19937 rng{time(NULL)};
    std::uniform_real_distribution<float> uniform{-1.0f, 1.0f};

    mNodeCount = nodeCount;
    mPrevCount = prevCount;
    mActivationFunction = ActivationFunction(activationFunction);
    mIsTraining = false;

    mBiases = std::vector<float>(nodeCount, 0.0f);
    mWeight = std::vector<std::vector<float> >(nodeCount, std::vector<float>(prevCount, 0.0f));

    for(int i = 0; i < nodeCount; i++){
      mBiases[i] = uniform(rng);
      for(int j = 0; j < prevCount; j++){
        mWeight[i][j] = uniform(rng);
      }
    }
  }

  NeuralNetworkLayer::~NeuralNetworkLayer(){
    if(mIsTraining){
      stopTraining();
    }
  }

  std::vector<float> NeuralNetworkLayer::feedForward(const std::vector<float>& inputData){

    std::vector<float> result{mNodeCount};

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

  void NeuralNetworkLayer::gradientFromExample(const std::vector<float>& exampleData) {
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

  void NeuralNetworkLayer::gradientFromActives(const std::vector<float>& inputData) {
    if(!mIsTraining)
      return;

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

    pValues = new std::vector<float>(mNodeCount, 0.0f);

    pValuesG = new std::vector<float>(mNodeCount, 0.0f);

    pBiasesGS = new std::vector<float>(mNodeCount, 0.0f);
    pWeightGS = new std::vector<std::vector<float> >(mNodeCount, std::vector<float>(mPrevCount, 0.0f));

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


} /* simplemind */
