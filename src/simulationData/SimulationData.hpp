#ifndef N_BODY_SIMULATION_SIMULATIONDATA_HPP
#define N_BODY_SIMULATION_SIMULATIONDATA_HPP

#include <string>
#include <array>
#include <vector>

/*
 * Struct which stores all the simulation data. Each vector contains all the entries of one specific attribute of the
 * simulation data. The id of one specific body corresponds to the index in the vector. E.g.: The attributes of the body
 * with id 0 are found at index 0 in the corresponding vectors.
 */
struct SimulationData {
    std::vector<std::string> names;
    std::vector<std::string> body_classes;
    std::vector<double> mass;

    // position data
    std::vector<double> positions_x;
    std::vector<double> positions_y;
    std::vector<double> positions_z;

    // velocity data
    std::vector<double> velocities_x;
    std::vector<double> velocities_y;
    std::vector<double> velocities_z;
};

#endif //N_BODY_SIMULATION_SIMULATIONDATA_HPP
