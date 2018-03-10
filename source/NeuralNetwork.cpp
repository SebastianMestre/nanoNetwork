#include "header/NeuralNetwork.hpp"

namespace simplemind {

  NeuralNetwork::NeuralNetwork(std::size_t inputCount, std::size_t outputCount):mInputCount(inputCount), mOutputCount(outputCount){
    mOutputLayer = NeuralNetworkLayer(outputCount, inputCount, ActivationFunction::Linear);
  }

  void NeuralNetwork::addLayer(std::size_t nodeCount, ActivationFunction::activationEnum activationFunction){
    if(mLayerCount == 0)
      mHiddenLayers.push_back(NeuralNetworkLayer(nodeCount, inputCount, activationFunction));
    else
      mHiddenLayers.push_back(NeuralNetworkLayer(nodeCount, mHiddenLayers.back().getNodeCount(), activationFunction));

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

  void NeuralNetwork::train( const std::vector<std::pair<std::vector<float>, std::vector<float> > >& trainData, std::size_t batchSize, std::size_t epochs, float learningRate){
    // TODO:
    // make static / global rng
    std::mt19937 rng{time(0)};
    int trainExamples = (int)trainData.size();

    for(int i = 0; i < mLayerCount; i++){
      mHiddenLayers[i].startTraining();
    }
    mOutputLayer.startTraining();

    for (int iEpoch = 0; iEpoch < epochs; iEpoch++) {
      std::random_shuffle(trainData.begin(), trainData.end(), rng);

      for(int k = 0; k < trainExamples; k += batchSize ){
        for(int iExample = k; iExample < k + batchSize && iExample < trainExamples; iExample++){

          process(trainData[iExample].first);

          mOutputLayer.gradientFromExample(trainData[iExample].second);
          if(mLayerCount == 0){
            mOutputLayer.gradientFromAnother(trainData[iExample].first);
          }else{
            mOutputLayer.gradientFromAnother(mHiddenLayers[mLayerCount-1]);
          }

          for(int i = mLayerCount-1; i >= 0; i--){

            if(i == mLayerCount-1){
              mHiddenLayers[i].gradientFromAnother(mOutputLayer);
            }else{
              mHiddenLayers[i].gradientFromAnother(mHiddenLayers[i+1]);
            }

            if(i == 0){
              mHiddenLayers[i].gradientFromActives(trainData[iExample].first);
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

} /* simplemind */
