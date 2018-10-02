#include "NeuralNetworkLayer.hpp"

namespace nanoNet {

    NeuralNetworkLayer::NeuralNetworkLayer (
        std::size_t inputCount,
        std::size_t outputCount,
        const ActivationFunction& activationFunction
    ) :
        m_inputCount (inputCount),
        m_outputCount (outputCount),
        m_activationFunction (activationFunction),
        m_biases (m_outputCount),
        m_weights (m_outputCount, vector_type(m_inputCount))
    {
        std::mt19937 rng(std::time(NULL));
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

        for ( auto& value : m_biases )
            value = uniform ( rng );

        for ( auto& values : m_weights )
            for ( auto& value : values )
                value = uniform ( rng );
    }

    NeuralNetworkLayer::vector_type NeuralNetworkLayer::predict (
        const vector_type& inputData
    ) const {
        vector_type result(m_biases);

        for (std::size_t i = 0; i < m_outputCount; i++)
            for (std::size_t j = 0; j < m_inputCount; j++)
                result[i] += m_weights[i][j] * inputData[j];

        for(auto& value : result)
            value = m_activationFunction(value);

        return result;
    }

} /* nanoNet */
