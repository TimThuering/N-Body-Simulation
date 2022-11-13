#ifndef N_BODY_SIMULATION_NAIVEALGORITHM_HPP
#define N_BODY_SIMULATION_NAIVEALGORITHM_HPP

#include "nBodyAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

class NaiveAlgorithm : public nBodyAlgorithm {
public:
    NaiveAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory,
                   std::size_t numberOfBodies);

    /*
     * Computes an n-body simulation with the naive algorithm
     */
    void startSimulation(const SimulationData &simulationData) override;

private:
    /*
     * computes the acceleration of each body
     */
    void computeAccelerations(queue &queue, buffer<double> &masses, std::vector<double> &currentPositions_x,
                              std::vector<double> &currentPositions_y, std::vector<double> &currentPositions_z,
                              std::vector<double> &acceleration_x, std::vector<double> &acceleration_y,
                              std::vector<double> &acceleration_z);

    /*
     * computes the kinetic, potential and total energy and the virial equilibrium in the current step and stores
     * the computed values.
     */
    void computeEnergy(const SimulationData &simulationData, std::size_t currentStep);

    /*
     * Compute the norm of the x,y and z component of the acceleration of each body in the current time step and saves
     * the values.
     */
    void storeAccelerations(std::size_t currentStep, std::vector<double> &acceleration_x,
                            std::vector<double> &acceleration_y, std::vector<double> &acceleration_z);

    /*
     * This function adjusts the initial velocity values and will be used once before the first simulation step
     */
    void adjustVelocities(const SimulationData &simulationData);
};

#endif //N_BODY_SIMULATION_NAIVEALGORITHM_HPP
