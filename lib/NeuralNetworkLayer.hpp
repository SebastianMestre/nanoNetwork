#ifndef NANONET_NEURALNETWORKLAYER_HPP
#define NANONET_NEURALNETWORKLAYER_HPP

#include <ctime>
#include <random>
#include <vector>

#include "ActivationFunction.hpp"

namespace nanoNet {

    class NeuralNetworkLayer {
        friend class Trainer;
    public:
        using real_type = float;
        using vector_type = std::vector<real_type>;
        using matrix_type = std::vector<vector_type>;
    private:
        std::size_t m_inputCount;
        std::size_t m_outputCount;

        ActivationFunction m_activationFunction;
        vector_type m_biases;
        matrix_type m_weights;

    public:
        NeuralNetworkLayer();

        NeuralNetworkLayer(
            std::size_t inputCount,
            std::size_t outputCount,
            const ActivationFunction& activationFunction
        );

        ~NeuralNetworkLayer() = default;

        vector_type predict (const vector_type& inputData) const;

        std::size_t inputCount () const {return m_inputCount;}
        std::size_t outputCount () const {return m_outputCount;}

        const vector_type& biases () const {return m_biases;}
        const matrix_type& weights () const {return m_weights;}
    };
} /* nanoNet */

#endif // NANONET_NEURALNETWORKLAYER_HPP
