#ifndef N_BODY_SIMULATION_CONFIGURATION_HPP
#define N_BODY_SIMULATION_CONFIGURATION_HPP

#include <cstddef>
#include <string>
#include <sycl/sycl.hpp>

namespace d_type {
    typedef unsigned int int_t;     // type definition for integer data type
}

namespace configuration {
    // total number of bodies used in the simulation
    extern d_type::int_t numberOfBodies;

    // Softening factor for the acceleration computation
    extern double epsilon2;

    // compute the energy of the system in each visualized time step.
    extern bool compute_energy;

    // If true, GPUs will be used for the computation. If false CPUs will be used.
    extern bool use_GPUs;

    namespace naive_algorithm {
        /*
         * Used by the naive algorithm.
         * the tile size determines the size of the blocks after which the local memory is updated again.
         * This parameter also implicitly defines the size of the work-groups used in the computeAccelerations kernel of the
         * naive algorithm.
         */
        extern int blockSize;

        // if true, the GPU optimized version of the acceleration kernel will be used. Otherwise, the version without the optimizations gets used
        extern bool GPU_Kernel;

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
         * Defines the work-group size for the acceleration kernel of the Barnes-Hut algorithm.
         */
        extern int workGroupSize;

        /*
         * Parameter that determines the size of the stack used to traverse the tree during the acceleration computation
         * of the Barnes-Hut algorithm.
         * Depends on the log_2 of the number of bodies in the example.
         */
        extern d_type::int_t stackSize;

        /*
         * Parameter used to determine the amount of storage that will be allocated for various fields needed by the
         * Barnes-Hut algorithm implementation.
         * The value will depend on the total number of bodies used for the simulation in order to scale with it.
         */
        extern d_type::int_t storageSizeParameter;

        // Number of work-items used for AABB computation
        extern int AABBWorkItemCount;

        // Number of work-items used for the octree creation
        extern int octreeWorkItemCount;

        // Number of work-items used for the creation of the top levels of the octree
        extern int octreeTopWorkItemCount;

        // Number of work-items used for the calculation of the center of mass for each node of the octree
        extern int centerOfMassWorkItemCount;

        /*
         * If ParallelOctreeTopDownSubtree is used, this parameter determines the maximum level to which the octree is
         * build in the first Phase.
         */
        extern int maxBuildLevel;

        /*
         * If true, sorting of the bodies according to their position in the tree gets enabled, which can
         * improve performance of the acceleration computation on GPUs
         */
        extern bool sortBodies;
    }


    /*
     * Initializes the number of bodies used for this simulation and all configuration values that depend on the number of bodies
     */
    void initializeConfigValues(d_type::int_t bodyCount, int storageSizeParam, int stackSizeParam);

    // setter for the configuration values

    void setBlockSize(int blockSize);

    void setTheta(double theta);

    void setAABBWorkItemCount(int workItemCount);

    void setOctreeWorkItemCount(int workItemCount);

    void setOctreeTopWorkItemCount(int workItemCount);

    void setCenterOfMassWorkItemCount(int workItemCount);

    void setMaxBuildLevel(int maxLevel);

    void setEnergyComputation(bool computeEnergy);

    void setSortBodies(bool sort_bodies);

    void setDeviceGPU(bool useGPU);

    void setWorkGroupSizeBarnesHut(int workGroupSize);

    void setUseGPUKernel(bool useGPUKernel);

}


#endif //N_BODY_SIMULATION_CONFIGURATION_HPP
