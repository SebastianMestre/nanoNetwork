#include <iostream>
#include <algorithm>
#include <random>
#include "../lib/NeuralNetwork.hpp"
#include "../lib/Trainer.hpp"

using namespace nanoNet;

int main() {

    NeuralNetwork network;

    network.addLayer({2, 8, {ActivationFunction::Which::Relu}});
    network.addLayer({8, 1, {ActivationFunction::Which::Linear}});

    // 10000 epochs, batch size = 2, training rate = 0.002
    Trainer trainer { 10000 , 2 , 0.002 };

    Trainer::DataSet data {{
        {{{0.0, 0.0}},{{0.0}}},
        {{{0.0, 1.0}},{{1.0}}},
        {{{1.0, 0.0}},{{1.0}}},
        {{{1.0, 1.0}},{{0.0}}}
    }};

    // stochastic gradient descent
    trainer.train ( network, data );

    auto rnd_base = []( float k ){ return [=]( float x ){ return ( int(x*k+0.5) ) / k; }; };
    auto rnd = rnd_base(2);

    std::cout << '\n';
    std::cout << rnd(network.predict({{0.0, 0.0}})[0]) << '\n';
    std::cout << rnd(network.predict({{0.0, 1.0}})[0]) << '\n';
    std::cout << rnd(network.predict({{1.0, 0.0}})[0]) << '\n';
    std::cout << rnd(network.predict({{1.0, 1.0}})[0]) << '\n';
    std::cout << '\n';

    return 0;
}
