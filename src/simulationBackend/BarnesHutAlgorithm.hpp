#ifndef N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP
#define N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP

#include "nBodyAlgorithm.hpp"
#include "BarnesHutOctree.hpp"
#include "ParallelOctreeTopDownSynchronized.hpp"
#include "ParallelOctreeTopDownSubtrees.hpp"
#include "SimulationData.hpp"
#include "Configuration.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;
#ifdef OCTREE_TOP_DOWN_SYNC
typedef ParallelOctreeTopDownSynchronized Octree;
#else
typedef ParallelOctreeTopDownSubtrees Octree;
#endif

class BarnesHutAlgorithm : public nBodyAlgorithm {
public:

    // Contains the octree data structure including an operation to build the octree.
    Octree octree;

    std::vector<std::size_t> nodesOnStack_vec;
    buffer<std::size_t> nodesOnStack;

    BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory);

    /*
     * Computes an n-body simulation with the Barnes-Hut algorithm
     */
    void startSimulation(const SimulationData &simulationData) override;

    /*
     * Computes the acceleration of each body induced by the gravitation of all the other bodies.
     * The functions makes use of SYCL for parallel execution.
     *
     * The masses buffer contains the masses of all the buffers.
     * The 3 buffers current_position_{x,y,z} contain the current position of all the bodies.
     * The 3 buffers acceleration_{x,y,z} will be used to store the computed accelerations.
     */
    void computeAccelerations(std::vector<queue> &queues, buffer<double> &masses, buffer<double> &currentPositions_x,
                              buffer<double> &currentPositions_y, buffer<double> &currentPositions_z,
                              buffer<double> &acceleration_x, buffer<double> &acceleration_y,
                              buffer<double> &acceleration_z);
};

#endif //N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP
