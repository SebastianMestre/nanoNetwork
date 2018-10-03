#include <iostream>
#include "../lib/NeuralNetwork.hpp"
#include "../lib/Trainer.hpp"

using namespace nanoNet;

int main() {

    NeuralNetwork pepe;

    pepe.addLayer({256, 16, {ActivationFunction::Which::Linear}});
    pepe.addLayer({16, 16, {ActivationFunction::Which::Linear}});
    pepe.addLayer({16, 10, {ActivationFunction::Which::Linear}});

    return 0;
}
