#ifndef N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP
#define N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP

#include "nBodyAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

class BarnesHutAlgorithm : public nBodyAlgorithm {
public:
    // minimum and maximum x coordinates of the axis-aligned bounding box (AABB) of the simulation data.
    // note that in this case the AABB is a cube with edge Length AABB_EdgeLength
    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    double AABB_EdgeLength;

    // data structures for linearized octree. The indices of the vectors correspond to the node IDs.

    // the upper 4 octants of one node in the tree
    std::vector<std::size_t> upper_NW;
    std::vector<std::size_t> upper_NE;
    std::vector<std::size_t> upper_SW;
    std::vector<std::size_t> upper_SE;

    // the lower 4 octants of one node in the tree
    std::vector<std::size_t> lower_NW;
    std::vector<std::size_t> lower_NE;
    std::vector<std::size_t> lower_SW;
    std::vector<std::size_t> lower_SE;

    // the edge length of the octants
    std::vector<double> edgeLengths;

    // together with the edge length, the min x,y,z values completely define the cube that represents the respective octant.
    std::vector<double> min_x_values;
    std::vector<double> min_y_values;
    std::vector<double> min_z_values;

    // stores a body ID if the node is a leaf node.
    // the ID numberOfBodies denotes that the node does not hold any bodies, i.e. is not a leaf node
    std::vector<std::size_t> bodyOfNode;

    // the sums of the masses of all bodies in the respective octants
    std::vector<double> sumMasses;

    // the center of the masses of all bodies in the respective octants
    std::vector<double> centerOfMass;

    BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory,
                       std::size_t numberOfBodies);

    /*
     * Computes an n-body simulation with the Barnes-Hut algorithm
     */
    void startSimulation(const SimulationData &simulationData) override;

    /*
     * This function builds an octree containing all bodies of the simulation
     */
    void buildOctree(queue &queue, std::vector<double> &current_positions_x, std::vector<double> &current_positions_y,
                     std::vector<double> &current_positions_z);


    /*
     * This function computes the minimum and maximum x,y,z values of the positions in the simulation data
     * and stores them in the corresponding class variables.
     */
    void computeMinMaxValuesAABB(queue &queue, const SimulationData &simulationData);

    /*
     * This function splits the node with NodeID into 8 octants.
     * The values of the new nodes will be stored at the positions following firstIndex.
     */
    void splitNode(std::size_t nodeID, std::size_t firstIndex);

    /*
     * Returns the node ID of the child octant of the parent node that would contain the body specified through its x,y,z position.
     */
    std::size_t getOctantContainingBody(double body_position_x, double body_position_y, double body_position_z,
                                        std::size_t parentNodeID);
    };

#endif //N_BODY_SIMULATION_BARNESHUTALGORITHM_HPP
