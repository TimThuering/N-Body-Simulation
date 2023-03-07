#ifndef N_BODY_SIMULATION_NAIVEALGORITHM_HPP
#define N_BODY_SIMULATION_NAIVEALGORITHM_HPP

#include "nBodyAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

/*
 * This class represents the naive algorithm for n-body simulations.
 */
class NaiveAlgorithm : public nBodyAlgorithm {
public:
    NaiveAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory);

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
     * Computes the acceleration of each body induced by the gravitation of all the other bodies.
     * In contrast to computeAccelerationsGPU it does not implement the optimizations for GPU execution.
     */
    void computeAccelerationsCPU(queue &queue, buffer<double> &masses, buffer<double> &currentPositions_x,
                                 buffer<double> &currentPositions_y, buffer<double> &currentPositions_z,
                                 buffer<double> &acceleration_x, buffer<double> &acceleration_y,
                                 buffer<double> &acceleration_z);
};

#endif //N_BODY_SIMULATION_NAIVEALGORITHM_HPP
