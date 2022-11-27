#include "BarnesHutAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>
#include "list"
#include <chrono>

using namespace sycl;

BarnesHutAlgorithm::BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth,
                                       std::string &outputDirectory, std::size_t numberOfBodies)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory, numberOfBodies) {
    this->description = "Barnes-Hut Algorithm";

}

void BarnesHutAlgorithm::startSimulation(const SimulationData &simulationData) {

    // SYCL queue for computation tasks
    queue queue;

}

void BarnesHutAlgorithm::buildOctree(queue &queue, std::vector<double> &current_positions_x,
                                     std::vector<double> &current_positions_y,
                                     std::vector<double> &current_positions_z) {

    std::list<std::size_t> currentLeafNodes;
    std::size_t nextFreeNodeID = 0;
    std::size_t currentBody = 0;

    // root node 0: the AABB of all bodies
    edgeLengths.push_back(AABB_EdgeLength);
    min_x_values.push_back(min_x);
    min_y_values.push_back(min_y);
    min_z_values.push_back(min_z);

    // root has no children yet.
    upper_NW.push_back(0);
    upper_NE.push_back(0);
    upper_SW.push_back(0);
    upper_SE.push_back(0);
    lower_NW.push_back(0);
    lower_NE.push_back(0);
    lower_SW.push_back(0);
    lower_SE.push_back(0);


    bodyOfNode.push_back(currentBody); // insert the first body into the root.
    currentLeafNodes.push_back(nextFreeNodeID); // currently, the node is a leaf node

    nextFreeNodeID += 1;

    for (std::size_t i = 1; i < numberOfBodies; ++i) {

        std::size_t currentNode = 0;
        bool nodeInserted = false;
        while (!nodeInserted) {
            if (upper_NW[currentNode] == 0) {
                // the current node is a leaf node
                if (bodyOfNode[currentNode] == numberOfBodies) {
                    // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body
                    bodyOfNode[currentNode] = i;
                    nodeInserted = true;
                } else {
                    // the leaf node already contains a body --> split the node and insert old body
                    std::size_t bodyIDinNode = bodyOfNode[currentNode];
                    splitNode(currentNode, nextFreeNodeID);
                    nextFreeNodeID += 8;

                    std::size_t octantID = getOctantContainingBody(current_positions_x[bodyIDinNode], current_positions_y[bodyIDinNode],
                                            current_positions_z[bodyIDinNode], currentNode);
                    // insert the old body into the new octant it belongs to and remove it from the parent node
                    bodyOfNode[octantID] = bodyIDinNode;
                    bodyOfNode[currentNode] = numberOfBodies;
                }
            } else {
                // the current node is not a leaf node, i.e. it has 8 children
                // --> determine the octant, the body has to be inserted and set this octant as current node.
                std::size_t octantID = getOctantContainingBody(current_positions_x[i], current_positions_y[i],
                                                               current_positions_z[i], currentNode);
                currentNode = octantID;
            }

        }
    }


}

void BarnesHutAlgorithm::computeMinMaxValuesAABB(queue &queue, const SimulationData &simulationData) {

    // update min and max values of the x,y,z coordinates
    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        double current_x = simulationData.positions_x[i];
        double current_y = simulationData.positions_y[i];
        double current_z = simulationData.positions_z[i];

        if (current_x < min_x) {
            min_x = current_x;
        }

        if (current_y < min_y) {
            min_y = current_y;
        }

        if (current_z < min_z) {
            min_z = current_z;
        }

        if (current_x > max_x) {
            max_x = current_x;
        }

        if (current_y > max_y) {
            max_y = current_y;
        }

        if (current_z > max_z) {
            max_z = current_z;
        }

    }

    // The AABB is now a square: transformation into cube
    double x_length = std::abs(max_x - min_x);
    double y_length = std::abs(max_y - min_y);
    double z_length = std::abs(max_z - min_z);
    double maxEdgeLength = std::max(x_length, std::max(y_length, z_length));

    // Extend the lengths of the shorter edges to maxEdgeLength by adding the difference equally to each side of the edge.
    if (maxEdgeLength == x_length) {
        min_z = min_z - ((maxEdgeLength - z_length) / 2);
        min_y = min_y - ((maxEdgeLength - y_length) / 2);

        max_z = max_z + ((maxEdgeLength - z_length) / 2);
        max_y = max_y + ((maxEdgeLength - y_length) / 2);
    } else if (maxEdgeLength == y_length) {
        min_x = min_x - ((maxEdgeLength - x_length) / 2);
        min_z = min_z - ((maxEdgeLength - z_length) / 2);

        max_x = max_x + ((maxEdgeLength - x_length) / 2);
        max_z = max_z + ((maxEdgeLength - z_length) / 2);
    } else if (maxEdgeLength == z_length) {
        min_x = min_x - ((maxEdgeLength - x_length) / 2);
        min_y = min_y - ((maxEdgeLength - y_length) / 2);

        max_x = max_x + ((maxEdgeLength - x_length) / 2);
        max_y = max_y + ((maxEdgeLength - y_length) / 2);
    } else {
        throw std::invalid_argument("maxEdgeLength has to be the x,y or z length");
    }

    AABB_EdgeLength = maxEdgeLength;


}

void BarnesHutAlgorithm::splitNode(std::size_t nodeID, std::size_t firstIndex) {

    double childEdgeLength = edgeLengths[nodeID] / 2;


    // set the edge lengths of the child nodes
    for (int i = 0; i < 8; ++i) {
        edgeLengths.push_back(childEdgeLength);
    }

    // create the 8 new octants
    upper_NW[nodeID] = firstIndex;
    upper_NE[nodeID] = firstIndex + 1;
    upper_SW[nodeID] = firstIndex + 2;
    upper_SE[nodeID] = firstIndex + 3;

    lower_NW[nodeID] = firstIndex + 4;
    lower_NE[nodeID] = firstIndex + 5;
    lower_SW[nodeID] = firstIndex + 6;
    lower_SE[nodeID] = firstIndex + 7;


    // min x,y,z values of the upperNW child node
    min_x_values.push_back(min_x);
    min_y_values.push_back(min_y + childEdgeLength);
    min_z_values.push_back(min_z);

    // min x,y,z values of the upperNE child node
    min_x_values.push_back(min_x + childEdgeLength);
    min_y_values.push_back(min_y + childEdgeLength);
    min_z_values.push_back(min_z);

    // min x,y,z values of the upperSW child node
    min_x_values.push_back(min_x);
    min_y_values.push_back(min_y + childEdgeLength);
    min_z_values.push_back(min_z + childEdgeLength);

    // min x,y,z values of the upperSE child node
    min_x_values.push_back(min_x + childEdgeLength);
    min_y_values.push_back(min_y + childEdgeLength);
    min_z_values.push_back(min_z + childEdgeLength);

    // min x,y,z values of the lowerNW child node
    min_x_values.push_back(min_x);
    min_y_values.push_back(min_y);
    min_z_values.push_back(min_z);

    // min x,y,z values of the lowerNE child node
    min_x_values.push_back(min_x + childEdgeLength);
    min_y_values.push_back(min_y);
    min_z_values.push_back(min_z);

    // min x,y,z values of the lowerSW child node
    min_x_values.push_back(min_x);
    min_y_values.push_back(min_y);
    min_z_values.push_back(min_z + childEdgeLength);

    // min x,y,z values of the lowerSE child node
    min_x_values.push_back(min_x + childEdgeLength);
    min_y_values.push_back(min_y);
    min_z_values.push_back(min_z + childEdgeLength);

    // initially, the newly created octants will not have any children.
    // 0 is also the root node index, but since the root will never be a child of any node, it can be used here to identify
    // leaf nodes.
    // Furthermore, since these nodes do not contain any bodies yet, the impossible body ID numberOfBodies gets used.
    for (int i = 0; i < 8; i++) {
        upper_NW.push_back(0);
        upper_NE.push_back(0);
        upper_SW.push_back(0);
        upper_SE.push_back(0);
        lower_NW.push_back(0);
        lower_NE.push_back(0);
        lower_SW.push_back(0);
        lower_SE.push_back(0);

        bodyOfNode.push_back(numberOfBodies);
    }


}
std::size_t
BarnesHutAlgorithm::getOctantContainingBody(double body_position_x, double body_position_y, double body_position_z,
                                            std::size_t parentNodeID) {
    double parentMin_x = min_x_values[parentNodeID];
    double parentMin_y = min_y_values[parentNodeID];
    double parentMin_z = min_z_values[parentNodeID];
    double parentEdgeLength = edgeLengths[parentNodeID];

    bool upperPart = body_position_y > parentMin_y + (parentEdgeLength / 2);
    bool rightPart = body_position_x > parentMin_x + (parentEdgeLength / 2);
    bool backPart = body_position_z < parentMin_z + (parentEdgeLength / 2);

    if (!upperPart && !rightPart && !backPart) {
        return lower_SW[parentNodeID];
    } else if (!upperPart && !rightPart && backPart) {
        return lower_NW[parentNodeID];
    } else if (!upperPart && rightPart && !backPart) {
        return lower_SE[parentNodeID];
    } else if (!upperPart && rightPart && backPart) {
        return lower_NE[parentNodeID];
    } else if (upperPart && !rightPart && !backPart) {
        return upper_SW[parentNodeID];
    } else if (upperPart && !rightPart && backPart) {
        return upper_NW[parentNodeID];
    } else if (upperPart && rightPart && !backPart) {
        return upper_SE[parentNodeID];
    } else if (upperPart && rightPart && backPart) {
        return upper_NE[parentNodeID];
    } else {
        throw std::runtime_error("This state should not be reachable");
    }
}
