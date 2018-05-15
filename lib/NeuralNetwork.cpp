#include "NeuralNetwork.hpp"

namespace nanoNet {

  NeuralNetwork::NeuralNetwork(std::size_t inputCount, std::size_t outputCount):
  outputLayer(outputCount, inputCount, ActivationFunction::Linear){
    this->inputCount = inputCount;
    this->outputCount = outputCount;
    layerCount = 0u;
  }

  void NeuralNetwork::addLayer(std::size_t nodeCount, ActivationFunction::activationEnum activationFunction){
    if(layerCount == 0){
      hiddenLayers.push_back(NeuralNetworkLayer(nodeCount, inputCount, activationFunction));
    }else{
      hiddenLayers.push_back(NeuralNetworkLayer(nodeCount, hiddenLayers.back().getNodeCount(), activationFunction));
    }

    outputLayer = NeuralNetworkLayer(outputCount, hiddenLayers.back().getNodeCount(), ActivationFunction::Linear);
  }

  std::vector<float> NeuralNetwork::process(const std::vector<float>& inputData){

    if(layerCount == 0)
      return outputLayer.feedForward(inputData);

    std::vector<float> v = inputData;
    for (int i = 0; i < layerCount; i++) {
      v = hiddenLayers[i].feedForward(v);
    }
    return outputLayer.feedForward(v);

  }

  void NeuralNetwork::train( const std::vector<DataPoint>& trainData, std::size_t batchSize, std::size_t epochs, float learningRate){

    int trainExamples = (int)trainData.size();

    for(int i = 0; i < layerCount; i++){
      hiddenLayers[i].startTraining();
    }
    outputLayer.startTraining();

    for (int iEpoch = 0; iEpoch < epochs; iEpoch++) {
      ///std::shuffle(trainData.begin(), trainData.end(), std::default_random_engine(0));

      for(int k = 0; k < trainExamples; k += batchSize ){
        for(int iExample = k; iExample < k + batchSize && iExample < trainExamples; iExample++){

          process(trainData[iExample].input);

          outputLayer.gradientFromExample(trainData[iExample].output);
          if(layerCount == 0){
            outputLayer.gradientFromActives(trainData[iExample].input);
          }else{
            outputLayer.gradientFromActives(hiddenLayers[layerCount-1]);
          }

          for(int i = layerCount-1; i >= 0; i--){

            if(i == layerCount-1){
              hiddenLayers[i].gradientFromAnother(outputLayer);
            }else{
              hiddenLayers[i].gradientFromAnother(hiddenLayers[i+1]);
            }

            if(i == 0){
              hiddenLayers[i].gradientFromActives(trainData[iExample].input);
            }else{
              hiddenLayers[i].gradientFromActives(hiddenLayers[i+1]);
            }
          }
        }

        for(int i = 0; i < layerCount; i++){
          hiddenLayers[i].applyTraining(learningRate, batchSize);
        }
        outputLayer.applyTraining(learningRate, batchSize);
      }
    }

    for(int i = 0; i < layerCount; i++){
      hiddenLayers[i].stopTraining();
    }
    outputLayer.stopTraining();
  }

} /* nanoNet */
