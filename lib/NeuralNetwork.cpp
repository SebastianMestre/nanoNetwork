#include "NeuralNetwork.hpp"

namespace nanoNet {

  NeuralNetwork::NeuralNetwork(std::size_t inputCount, std::size_t outputCount):
  mOutputLayer(outputCount, inputCount, ActivationFunction::Linear){
    mInputCount = inputCount;
    mOutputCount = outputCount;
    mLayerCount = 0u;
  }

  void NeuralNetwork::addLayer(std::size_t nodeCount, ActivationFunction::activationEnum activationFunction){
    if(mLayerCount == 0){
      mHiddenLayers.push_back(NeuralNetworkLayer(nodeCount, mInputCount, activationFunction));
    }else{
      mHiddenLayers.push_back(NeuralNetworkLayer(nodeCount, mHiddenLayers.back().getNodeCount(), activationFunction));
    }

    mOutputLayer = NeuralNetworkLayer(mOutputCount, mHiddenLayers.back().getNodeCount(), ActivationFunction::Linear);
  }

  std::vector<float> NeuralNetwork::process(const std::vector<float>& inputData){

    if(mLayerCount == 0)
      return mOutputLayer.feedForward(inputData);

    std::vector<float> v = inputData;
    for (int i = 0; i < mLayerCount; i++) {
      v = mHiddenLayers[i].feedForward(v);
    }
    return mOutputLayer.feedForward(v);

  }

  void NeuralNetwork::train( const std::vector<DataPoint>& trainData, std::size_t batchSize, std::size_t epochs, float learningRate){

    int trainExamples = (int)trainData.size();

    for(int i = 0; i < mLayerCount; i++){
      mHiddenLayers[i].startTraining();
    }
    mOutputLayer.startTraining();

    for (int iEpoch = 0; iEpoch < epochs; iEpoch++) {
      ///std::shuffle(trainData.begin(), trainData.end(), std::default_random_engine(0));

      for(int k = 0; k < trainExamples; k += batchSize ){
        for(int iExample = k; iExample < k + batchSize && iExample < trainExamples; iExample++){

          process(trainData[iExample].input);

          mOutputLayer.gradientFromExample(trainData[iExample].output);
          if(mLayerCount == 0){
            mOutputLayer.gradientFromActives(trainData[iExample].input);
          }else{
            mOutputLayer.gradientFromActives(mHiddenLayers[mLayerCount-1]);
          }

          for(int i = mLayerCount-1; i >= 0; i--){

            if(i == mLayerCount-1){
              mHiddenLayers[i].gradientFromAnother(mOutputLayer);
            }else{
              mHiddenLayers[i].gradientFromAnother(mHiddenLayers[i+1]);
            }

            if(i == 0){
              mHiddenLayers[i].gradientFromActives(trainData[iExample].input);
            }else{
              mHiddenLayers[i].gradientFromActives(mHiddenLayers[i+1]);
            }
          }
        }

        for(int i = 0; i < mLayerCount; i++){
          mHiddenLayers[i].applyTraining(learningRate, batchSize);
        }
        mOutputLayer.applyTraining(learningRate, batchSize);
      }
    }

    for(int i = 0; i < mLayerCount; i++){
      mHiddenLayers[i].stopTraining();
    }
    mOutputLayer.stopTraining();
  }

} /* nanoNet */
