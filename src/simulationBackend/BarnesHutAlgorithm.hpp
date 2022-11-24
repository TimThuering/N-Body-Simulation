#ifndef N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP
#define N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP

#include "nBodyAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

class BarnesHutAlgorithm : public nBodyAlgorithm {
public:
    BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory,
                   std::size_t numberOfBodies);

    /*
     * Computes an n-body simulation with the naive algorithm
     */
    void startSimulation(const SimulationData &simulationData) override;




};

#endif //N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP
