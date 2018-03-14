#include <iostream>
#include "../lib/NeuralNetwork.hpp"

using namespace nanoNet;

int main() {
  std::cout << "Hello, World!" << std::endl;

  NeuralNetwork myNN{1, 1};

  myNN.addLayer(3, ActivationFunction::Tanh);

  return 0;
}
