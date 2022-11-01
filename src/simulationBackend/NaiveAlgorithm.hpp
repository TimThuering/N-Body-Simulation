#ifndef N_BODY_SIMULATION_NAIVEALGORITHM_HPP
#define N_BODY_SIMULATION_NAIVEALGORITHM_HPP

#include "nBodyAlgorithm.hpp"
#include "SimulationData.hpp"

class NaiveAlgorithm : public nBodyAlgorithm {
public:
    NaiveAlgorithm();

    /*
     * Computes an n-body simulation with the naive algorithm
     */
    void startSimulation(const SimulationData &simulationData) override;
};

#endif //N_BODY_SIMULATION_NAIVEALGORITHM_HPP
