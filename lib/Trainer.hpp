#pragma once

#include <vector>

#include "NeuralNetwork.hpp"
#include "NeuralNetworkLayer.hpp"

namespace nanoNet {

    struct DataPoint {
        std::vector<float> input;
        std::vector<float> output;
    };


    // Implementacion del patron de diseno "Visitor"
    class Trainer {
        // TODO: settings

        std::size_t epoch_count;
        std::size_t batch_size;
        float learning_rate;

    public:
        using DataSet = std::vector<DataPoint>;
        Trainer () {}

        void train (
            NeuralNetwork& network,
            DataSet dataSet
        ) const {

            if (network.layerCount() == 0)
                return;

            for (std::size_t iEpoch = 0; iEpoch < epoch_count; iEpoch++) {

                std::shuffle(dataSet.begin(), dataSet.end(), std::default_random_engine(0));

                for (std::size_t iDatum = 0; iDatum < dataSet.size();) {
                    for (std::size_t iBatch = 0; iBatch < batch_size && iDatum < dataSet.size(); ++iBatch, ++iDatum) {

                        std::vector<NeuralNetworkLayer::vector_type> results;

                        for (const auto& layer : network.m_layers)
                                results.push_back( layer.predict(
                                    results.empty()
                                        ? dataSet[iDatum].input
                                        : results.back()));

                        std::vector<NeuralNetworkLayer::vector_type> gradients(network.layerCount());

                        gradients.back() = gradientFromExample(
                            network.m_layers.back(),
                            results.back(),
                            dataSet[iDatum].output
                        );

                        for(int i = network.layerCount()-2; i >= 1; i--){
                            gradients[i] = gradientFromNext(
                                network.m_layers[i],
                                network.m_layers[i+1],
                                results[i+1],
                                gradients[i+1]
                            );

                            // network.hiddenLayers[i].gradientFromActives(network.hiddenLayers[i-1]);
                        }

                        // network.hiddenLayers[0].gradientFromActives(trainData[iDatum].input);

                    }

                    // for (auto& layer : network.hiddenLayers)
                        // layer.applyTraining(learningRate, batchSize);
                }
            }
        }

        static NeuralNetworkLayer::vector_type gradientFromExample (
            const NeuralNetworkLayer& layer,
            const NeuralNetworkLayer::vector_type& values,
            const NeuralNetworkLayer::vector_type& example
        ) {
            NeuralNetworkLayer::vector_type gradient ( layer.outputCount(), 0.0f );

            for (std::size_t i = 0; i < layer.outputCount(); ++i)
                gradient[i] = 2 * ( values[i] - example[i] );

            return gradient;
        }

        static NeuralNetworkLayer::vector_type gradientFromNext (
            const NeuralNetworkLayer& layer,
            const NeuralNetworkLayer& next,
            const NeuralNetworkLayer::vector_type& next_values,
            const NeuralNetworkLayer::vector_type& next_gradient
        ) {
            NeuralNetworkLayer::vector_type gradient ( layer.outputCount(), 0.0f );

            for (std::size_t i = 0; i < next.outputCount(); ++i)
                for (std::size_t j = 0; j < layer.outputCount(); ++j)
                    gradient[j] += next_gradient[i] * next.m_activationFunction[ next_values[i] ] * next.weights()[i][j];

            return gradient;
        }

    };

}
