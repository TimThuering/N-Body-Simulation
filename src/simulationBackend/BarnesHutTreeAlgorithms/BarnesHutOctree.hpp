#ifndef N_BODY_SIMULATION_BARNESHUTOCTREE_HPP
#define N_BODY_SIMULATION_BARNESHUTOCTREE_HPP


#include <vector>
#include <sycl/sycl.hpp>
#include "Configuration.hpp"

using namespace sycl;
using namespace configuration;

/*
 * Class from which all classes containing an implementation of an octree creation algorithm for the Barnes-Hut algorithm
 * are derived from.
 * Subclasses have to implement the function buildOctree();
 */
class BarnesHutOctree {
public:
    // minimum and maximum x coordinates of the axis-aligned bounding box (AABB) of the simulation data.
    // note that in this case the AABB is a cube with edge Length AABB_EdgeLength
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    double AABB_EdgeLength = 0;

    std::size_t maxTreeDepth = 0;

    // data structures for linearized octree:

    /*
     * Storage for all octants of the tree
     * This field can be imagined as 8 fields, each of size configuration::barnes_hut_algorithm::storageSizeParameter, right after each other.
     * The sequence is as follows:
     * LowerSW | LowerNW | LowerSE | LowerNE | UpperSW | UpperNW | UpperSE | UpperNE
     * The values correspond to nodeIDs of the respective octants. The index corresponds to the nodeID of the parent node
     * Example:
     * The nodeIDs of the octants of the root node with ID 0 can be found at
     * octants[0], octants[0 + 1 * size], octants[0 + 2 * size], ...
     * where octants[0] would contain the node ID of the lower south-west octant of the root node and size would
     * correspond to configuration::barnes_hut_algorithm::storageSizeParameter.
     */
    std::vector<std::size_t> octants_vec;
    buffer<std::size_t> octants;

    // the edge length of the octants
    std::vector<double> edgeLengths_vec;
    buffer<double> edgeLengths;

    // together with the edge length, the min x,y,z values completely define the cube that represents the respective octant.
    std::vector<double> min_x_values_vec;
    std::vector<double> min_y_values_vec;
    std::vector<double> min_z_values_vec;
    buffer<double> min_x_values;
    buffer<double> min_y_values;
    buffer<double> min_z_values;

    // stores a body ID if the node is a leaf node.
    // the ID numberOfBodies denotes that the node does not hold any bodies, i.e. is not a leaf node
    std::vector<std::size_t> bodyOfNode_vec;
    buffer<std::size_t> bodyOfNode;

    // storage for flags indicating that the corresponding node is a leaf node
    std::vector<int> nodeIsLeaf_vec;
    buffer<int> nodeIsLeaf;

    // the sums of the masses of all bodies in the respective octants
    std::vector<double> sumMasses_vec;
    buffer<double> sumOfMasses;

    // storage that will be used to atomically determine the next unused node ID
    std::vector<std::size_t> nextFreeNodeID_vec;
    buffer<std::size_t> nextFreeNodeID;

    // nodes for which the center of Mass has to be computed after the preparation steps and the first iteration
    std::vector<std::size_t> nodesToProcessCenterOfMass_vec;
    buffer<std::size_t> nodesToProcessCenterOfMass;

    // the center of the masses of all bodies in the respective octants (only contain the numerator --> have to be divided by the corresponding sum of masses)
    std::vector<double> centerOfMass_x_vec;
    std::vector<double> centerOfMass_y_vec;
    std::vector<double> centerOfMass_z_vec;
    buffer<double> massCenters_x;
    buffer<double> massCenters_y;
    buffer<double> massCenters_z;

    /*
     * constructor that initializes all buffers needed for a generic octree. Subclasses of this class might create more buffers.
     */
    BarnesHutOctree();

    virtual void buildOctree(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                             buffer<double> &current_positions_z, buffer<double> &masses) = 0;

    /*
     * This function computes the minimum and maximum x,y,z values of the positions in the simulation data in parallel
     * and stores them in the corresponding class variables.
     * The computation happens in parallel and uses SYCL for parallelization.
     */
    void computeMinMaxValuesAABB(queue &queue, buffer<double> &current_positions_x,
                                 buffer<double> &current_positions_y,
                                 buffer<double> &current_positions_z);

    /*
     * Computes the center of mass of all bodies in a cell of the octree. A cell in the octree is represented by a node.
     * This version can be used on GPUs.
     */
    void computeCenterOfMass_GPU(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                                 buffer<double> &current_positions_z, buffer<double> &masses);

    /*
     * Computes the center of mass of all bodies in a cell of the octree. A cell in the octree is represented by a node.
     * This version can be used on CPUs.
     */
    void computeCenterOfMass_CPU(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                                 buffer<double> &current_positions_z, buffer<double> &masses);

    /*
     * This function prepares the computation of the computeCenterOfMass functions in order to speed them up.
     * It computes the values of all nodes that are leaf nodes.
     */
    void prepareCenterOfMass(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                             buffer<double> &current_positions_z, buffer<double> &masses);
};


#endif //N_BODY_SIMULATION_BARNESHUTOCTREE_HPP
