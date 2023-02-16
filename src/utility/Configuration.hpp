#ifndef N_BODY_SIMULATION_CONFIGURATION_HPP
#define N_BODY_SIMULATION_CONFIGURATION_HPP

#include <cstddef>
#include <string>
#include <sycl/sycl.hpp>

namespace configuration {
    // total number of bodies used in the simulation
    extern std::size_t numberOfBodies;

    // Softening factor for the acceleration computation
    extern double epsilon2;

    // compute the energy of the system in each visualized time step.
    extern bool compute_energy;

    namespace naive_algorithm {
        /*
         * Used by the naive algorithm.
         * the tile size determines the size of the blocks after which the local memory is updated again.
         * This parameter also implicitly defines the size of the work-groups used in the computeAccelerations kernel of the
         * naive algorithm.
         * Value should be of the form 2^x.
         */
        extern int tileSizeNaiveAlg;

    }

    namespace barnes_hut_algorithm {
        /*
         * The theta parameter of the Barnes-Hut algorithm that gets used to determine when bodies in an octant
         * should be combined which simplifies the computation.
         * Lower values will result into better precision but worse performance whereas higher values will improve
         * performance but lose precision.
         */
        extern double theta;

        /*
         * Parameter that determines the size of the stack used to traverse the tree during the acceleration computation
         * of the Barnes-Hut algorithm.
         * Depends on the log_2 of the number of bodies in the example.
         */
        extern std::size_t stackSize;

        /*
         * Parameter used to determine the amount of storage that will be allocated for various fields needed by the
         * Barnes-Hut algorithm implementation.
         * The value will depend on the total number of bodies used for the simulation in order to scale with it.
         */
        extern std::size_t storageSizeParameter;

        // Number of work-items used for AABB computation
        extern int AABBWorkItemCount;

        // Number of work-items used for the octree creation
        extern int octreeWorkItemCount;

        // Number of work-items used for the creation of the top levels of the octree
        extern int octreeTopWorkItemCount;

        /*
         * If ParallelOctreeTopDownSubtree is used, this parameter determines the maximum level to which the octree is
         * build in the first Phase.
         */
        extern int maxBuildLevel;
    }


    /*
     * Initializes the number of bodies used for this simulation and all configuration values that depend on the number of bodies
     */
    void initializeConfigValues(std::size_t bodyCount, int storageSizeParam, int stackSizeParam);

    // setter for the configuration values

    void setBlockSize(int blockSize);

    void setTheta(double theta);

    void setAABBWorkItemCount(int workItemCount);

    void setOctreeWorkItemCount(int workItemCount);

    void setOctreeTopWorkItemCount(int workItemCount);

    void setMaxBuildLevel(int maxLevel);

    void setEnergyComputation(bool computeEnergy);

}

#endif //N_BODY_SIMULATION_CONFIGURATION_HPP
