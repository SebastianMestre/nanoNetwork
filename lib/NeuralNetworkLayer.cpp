#include "NeuralNetworkLayer.hpp"

namespace nanoNet {

  NeuralNetworkLayer::NeuralNetworkLayer(std::size_t nodeCount, std::size_t prevCount, ActivationFunction::activationEnum activationFunction)
  : mActivationFunction(activationFunction) {

    // std::mt19937 rng(std::time(NULL));
    // std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    this->nodeCount = nodeCount;
    this->prevCount = prevCount;
    isTraining = false;

    biases = std::vector<float>(nodeCount, 0.0f);
    weight = std::vector<std::vector<float> >(nodeCount, std::vector<float>(prevCount, 0.0f));

    for(int i = 0; i < nodeCount; i++){
      // biases[i] = uniform(rng);
      biases[i] = 0.0f;
      for(int j = 0; j < prevCount; j++){
        // weight[i][j] = uniform(rng);
        weight[i][j] = 0.0f;
      }
    }
  }

  NeuralNetworkLayer::~NeuralNetworkLayer(){
    // TODO: complete this
    if(isTraining){
      stopTraining();
    }
  }

  std::vector<float> NeuralNetworkLayer::feedForward(const std::vector<float>& inputData){

    std::vector<float> result(nodeCount);

    for (int i = 0; i < nodeCount; i++) {
      result[i] = biases[i];
      for (int j = 0; j < prevCount; j++) {
        result[i] += weight[i][j] * inputData[j];
      }
      result[i] = mActivationFunction(result[i]);
    }

    if(isTraining){
      (*pValues) = result;
    }

    return result;
  }

  void NeuralNetworkLayer::gradientFromExample(const std::vector<float>& exampleData) {
    if(!isTraining)
      return;

    for (int i = 0; i < nodeCount; i++) {
      (*pValuesG)[i] = 2 * ((*pValues)[i] - exampleData[i]);
    }
  }

  void NeuralNetworkLayer::gradientFromAnother(const NeuralNetworkLayer& next) {
    if(!isTraining)
      return;

    auto nextWeight = next.getWeight();
    for (size_t i = 0; i < nodeCount; i++) {
      (*pValuesG)[i] = 0.0f;
      for (int j = 0; j < next.nodeCount; j++) {
        (*pValuesG)[i] += (*next.pValuesG)[j] * next.mActivationFunction[ (*next.pValues)[j] ] * nextWeight[j][i];
      }

    }
  }

  void NeuralNetworkLayer::gradientFromActives(const NeuralNetworkLayer& prev) {
    gradientFromActives(*prev.pValues);
  }

  void NeuralNetworkLayer::gradientFromActives(const std::vector<float>& inputData) {
    if(!isTraining)
      return;

    // TODO: fix names
    for(int i = 0; i < nodeCount; i++){
      float aaa = (*pValuesG)[i] * mActivationFunction[ (*pValues)[i] ];

      (*pBiasesGS)[i] += aaa;

      for (int j = 0; j < prevCount; j++) {

        (*pWeightGS)[i][j] += aaa * inputData[j];

      }
    }
  }

  void NeuralNetworkLayer::substractGradients(float amount){
    if(!isTraining)
      return;

    for (int i = 0; i < nodeCount; i++) {
      biases[i] -= (*pBiasesGS)[i] * amount;
      for (int j = 0; j < prevCount; j++) {
        weight[i][j] -= (*pWeightGS)[i][j] * amount;
      }
    }
  }

  void NeuralNetworkLayer::startTraining() {
    if(!isTraining)
      return;

    pValues = new std::vector<float>(nodeCount, 0.0f);

    pValuesG = new std::vector<float>(nodeCount, 0.0f);

    pBiasesGS = new std::vector<float>(nodeCount, 0.0f);
    pWeightGS = new std::vector<std::vector<float> >(nodeCount, std::vector<float>(prevCount, 0.0f));

    isTraining = true;
  }

  void NeuralNetworkLayer::stopTraining() {
    if(!isTraining)
      return;

    delete pValues;

    delete pValuesG;

    delete pBiasesGS;
    delete pWeightGS;

    isTraining = false;
  }

  void NeuralNetworkLayer::applyTraining(float learningRate, int exampleCount){
    substractGradients(learningRate * exampleCount);

    std::fill(pBiasesGS->begin(), pBiasesGS->end(), 0.0f);
    for(auto& v : *pWeightGS){
      std::fill(v.begin(), v.end(), 0.0f);
    }
  }


} /* nanoNet */
