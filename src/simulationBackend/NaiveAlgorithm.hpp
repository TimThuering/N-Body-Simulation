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
     * Computes the acceleration of each body induced by the gravitation of all the other bodies.
     * The functions makes use of SYCL and is optimized for execution on GPUs.
     *
     * The masses buffer contains the masses of all the buffers.
     * The 3 buffers current_position_{x,y,z} contain the current position of all the bodies.
     * The 3 buffers acceleration_{x,y,z} will be used to store the computed accelerations.
     */
    void computeAccelerationsGPU(queue &queue, buffer<double> &masses, buffer<double> &currentPositions_x,
                                 buffer<double> &currentPositions_y, buffer<double> &currentPositions_z,
                                 buffer<double> &acceleration_x, buffer<double> &acceleration_y,
                                 buffer<double> &acceleration_z);


    /*
     * Computes the kinetic, potential and total energy and the virial equilibrium in the current step and stores
     * the computed values for the ParaView output.
     *
     * The function makes use of SYCL for parallel computation.
     * The 7 buffers masses, current_position_{x,y,z} and velocities_{x,y,z} contain the respective values of each body in the current time step.
     */
    void computeEnergy(queue &queue, buffer<double> &masses, std::size_t currentStep,
                       buffer<double> &currentPositions_x,
                       buffer<double> &currentPositions_y, buffer<double>
                       &currentPositions_z,
                       buffer<double> &velocities_x, buffer<double>
                       &velocities_y,
                       buffer<double> &velocities_z
    );

    /*
     * Compute the norm of the x,y and z component of the acceleration of each body in the current time step and saves
     * the values.
     */
    void storeAccelerations(std::size_t currentStep, buffer<double> &acceleration_x,
                            buffer<double> &acceleration_y, buffer<double> &acceleration_z);

    /*
     * This function adjusts the initial velocity values and will be used once before the first simulation step
     */
    void adjustVelocities(const SimulationData &simulationData);


};

#endif //N_BODY_SIMULATION_NAIVEALGORITHM_HPP
