#ifndef N_BODY_SIMULATION_PARALLELOCTREETOPDOWNSYNCHRONIZED_HPP
#define N_BODY_SIMULATION_PARALLELOCTREETOPDOWNSYNCHRONIZED_HPP

#include "BarnesHutOctree.hpp"


/*
 * This class contains an implementation of an octree which gets created in parallel using SYCL.
 * The approach is a synchronized parallel insertion of all bodies into the tree.
 */
class ParallelOctreeTopDownSynchronized : public BarnesHutOctree {
public:

    // storage for flags indicating that the corresponding node is currently locked, i.e. some thread currently is inserting a body.
    std::vector<int> nodeIsLocked_vec;
    buffer<int> nodeIsLocked;



    /*
     * This function implements the octree creation algorithm which follows a synchronized parallel insertion approach
     * and uses SYCL for parallelization
     */
    void buildOctree(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                     buffer<double> &current_positions_z, buffer<double> &masses) override;

    void computeCenterOfGravity(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                                 buffer<double> &current_positions_z, buffer<double> &masses);

    ParallelOctreeTopDownSynchronized();
};


#endif //N_BODY_SIMULATION_PARALLELOCTREETOPDOWNSYNCHRONIZED_HPP
