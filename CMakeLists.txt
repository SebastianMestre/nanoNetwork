cmake_minimum_required(VERSION 3.9)
project(nanoNetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(nanoNetwork testSrc/main.cpp
        lib/NeuralNetwork.cpp
        lib/ActivationFunction.cpp
        lib/NeuralNetworkLayer.cpp
        lib/ActivationFunction.hpp
        lib/NeuralNetwork.hpp
        lib/NeuralNetworkLayer.hpp)