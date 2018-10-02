#include "NeuralNetwork.hpp"

namespace nanoNet {

    NeuralNetwork::NeuralNetwork () :
        hiddenLayers()
    {}

    void NeuralNetwork::addLayer (
        const NeuralNetworkLayer& layer
    ) {
        if (layerCount() == 0 || outputCount() == layer.inputCount())
            hiddenLayers.push_back(layer);
    }

    NeuralNetworkLayer::vector_type NeuralNetwork::predict(
        const NeuralNetworkLayer::vector_type& inputData
    ) const {
        NeuralNetworkLayer::vector_type result(inputData);

        for (auto& layer : hiddenLayers)
            result = layer.predict(result);

        return result;
    }

} /* nanoNet */
