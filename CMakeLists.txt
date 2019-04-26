cmake_minimum_required(VERSION 3.9)
project(nanoNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(nanoNetwork example/main.cpp
        src/NeuralNetwork.cpp
        src/ActivationFunction.cpp
        src/NeuralNetworkLayer.cpp
        src/ActivationFunction.hpp
        src/NeuralNetwork.hpp
        src/NeuralNetworkLayer.hpp)
