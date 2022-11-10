#include <iostream>
#include "NaiveAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

NaiveAlgorithm::NaiveAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory,
                               std::size_t numberOfBodies)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory, numberOfBodies) {
    this->description = "Naive Algorithm";
}

void NaiveAlgorithm::startSimulation(const SimulationData &simulationData) {
    
}
