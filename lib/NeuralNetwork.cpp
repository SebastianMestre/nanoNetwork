#include "NeuralNetwork.hpp"

namespace nanoNet {

    NeuralNetwork::NeuralNetwork () :
        m_layers()
    {}

    void NeuralNetwork::addLayer (
        const NeuralNetworkLayer& layer
    ) {
        if (layerCount() == 0 || outputCount() == layer.inputCount())
            m_layers.push_back(layer);
    }

    NeuralNetworkLayer::vector_type NeuralNetwork::predict(
        const NeuralNetworkLayer::vector_type& inputData
    ) const {
        NeuralNetworkLayer::vector_type result(inputData);

        for (auto& layer : m_layers)
            result = layer.predict(result);

        return result;
    }

} /* nanoNet */
