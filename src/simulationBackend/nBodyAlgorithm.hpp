#ifndef N_BODY_SIMULATION_NBODYALGORITHM_HPP
#define N_BODY_SIMULATION_NBODYALGORITHM_HPP

#include <string>

class nBodyAlgorithm {
public:
    std::string description;
    virtual void startSimulation(struct simulationData) = 0;
};

#endif //N_BODY_SIMULATION_NBODYALGORITHM_HPP
