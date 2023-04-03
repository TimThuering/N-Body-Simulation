#ifndef N_BODY_SIMULATION_PARALLELOCTREETOPDOWNSUBTREES_HPP
#define N_BODY_SIMULATION_PARALLELOCTREETOPDOWNSUBTREES_HPP

#include "BarnesHutOctree.hpp"
#include "TimeMeasurement.hpp"

class ParallelOctreeTopDownSubtrees : public BarnesHutOctree {
public:
    // storage for flags indicating that the corresponding node is currently locked, i.e. some thread is currently inserting a body.
    std::vector<int> nodeIsLocked_vec;
    buffer<int> nodeIsLocked;

    // stores the nodeID of the root of the subtree each body belongs to
    std::vector<d_type::int_t> subtreeOfBody_vec;
    buffer<d_type::int_t> subtreeOfBody;

    // stores all bodies that still need to be inserted into the subtrees ordered by their corresponding subtrees
    std::vector<d_type::int_t> sortedBodies_vec;
    buffer<d_type::int_t> sortedBodies;

    // stores the amount of subtrees that were generated by buildOctreeToLevel()
    std::vector<d_type::int_t> subtreeCount_vec;
    buffer<d_type::int_t> subtreeCount;

    // stores the number of subtrees that emerged after building the top of the tree
    d_type::int_t numberOfSubtrees = 0;

    ParallelOctreeTopDownSubtrees();

    /*
     * This function implements the octree creation algorithm which follows a synchronized parallel insertion approach
     * and uses SYCL for parallelization. In contrast to ParallelOctreeTopDownSynchronized, it first builds the octree to
     * a certain level and determines the subtree of each body. After that it builds each subtree in parallel completely
     * independent of the other subtrees.
     */
    void buildOctree(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                     buffer<double> &current_positions_z, buffer<double> &masses, TimeMeasurement &timer) override;

    /*
     * This function builds the octree up to a certain level and determines for each body, into which subtree it belongs.
     * The level is determined by configuration::barnes_hut_algorithm::maxBuildLevel.
     * If a body can be inserted to its final position at a level < maxBuildLevel it will reference the root node as its subtree.
     * This indicates that the body has reached its final position and does not have to inserted into a subtree.
     */
    void buildOctreeToLevel(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                            buffer<double> &current_positions_z, buffer<double> &masses);

    /*
     * This function will be executed after the top of the tree has been build. It first iterates over all bodies and
     * determines which nodes are referencing a node of the tree as the root node of their subtree. In such a case
     * the amount of bodies in this subtree is incremented by one.
     * After that, in a second step, all nodes that represent subtree root nodes are extracted and stored in the separate
     * subtrees buffer.
     */
    void prepareSubtrees(queue &queue, buffer<d_type::int_t> &bodyCountSubtree, buffer<d_type::int_t> &subtrees,
                         d_type::int_t nodeCount);

    /*
     * This function sorts all bodies according to their subtrees. This step is needed for buildSubtrees(). It ensures
     * that all bodies that will be assigned to one work-group (i.e. belong to the same subtree) are stored right next to each
     * other. The start index of these bodies will be stored in bodiesOfSubtreeStartIndex.
     */
    void sortBodiesForSubtrees(queue &queue, buffer<d_type::int_t> &bodyCountSubtree, buffer<d_type::int_t> &subtrees,
                               buffer<d_type::int_t> &bodiesOfSubtreeStartIndex);

    /*
     * This function builds the subtrees which were determined in Phase 1 with buildOctreeToLevel().
     * Each subtree is build in parallel with a synchronized top-down approach.
     * The individual subtrees are completely independent of each other and are also build parallel to the other subtrees.
     */
    void buildSubtrees(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                       buffer<double> &current_positions_z, buffer<double> &masses,
                       buffer<d_type::int_t> &bodiesOfSubtreeStartIndex, buffer<d_type::int_t> &bodyCountSubtree,
                       buffer<d_type::int_t> &subtrees);
};


#endif //N_BODY_SIMULATION_PARALLELOCTREETOPDOWNSUBTREES_HPP
