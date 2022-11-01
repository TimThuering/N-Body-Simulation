#ifndef N_BODY_SIMULATION_NBODYALGORITHM_HPP
#define N_BODY_SIMULATION_NBODYALGORITHM_HPP

#include <string>
#include <map>
#include "SimulationData.hpp"

/*
 * Class from which all classes containing an implementation of an n-body algorithm are derived from.
 */
class nBodyAlgorithm {
public:
    std::string description; // The name of the algorithm

    // maps simulation step to computed position values
    std::map<std::size_t, std::vector<double>> positions_x;
    std::map<std::size_t, std::vector<double>> positions_y;
    std::map<std::size_t, std::vector<double>> positions_z;

    // maps simulation step to computed velocity values
    std::map<std::size_t, std::vector<double>> velocities_x;
    std::map<std::size_t, std::vector<double>> velocities_y;
    std::map<std::size_t, std::vector<double>> velocities_z;

    // maps simulation step to computed acceleration values
    std::map<std::size_t, std::vector<double>> acceleration;


     /*
     * Function that starts the actual computation of the n-body simulation
     * Has to be implemented in the subclass and will execute the corresponding algorithm.
     */
    virtual void startSimulation(const SimulationData &simulationData) = 0;

    /*
     * Generates all files necessary for the Paraview visualization
     */
    void generateParaViewOutput(const SimulationData &simulationData);

};

#endif //N_BODY_SIMULATION_NBODYALGORITHM_HPP
