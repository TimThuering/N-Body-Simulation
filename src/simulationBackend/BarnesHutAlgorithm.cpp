#include "BarnesHutAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

BarnesHutAlgorithm::BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth,
                                       std::string &outputDirectory, std::size_t numberOfBodies)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory, numberOfBodies) {
    this->description = "Barnes-Hut Algorithm";

}

void BarnesHutAlgorithm::startSimulation(const SimulationData &simulationData) {

}


