#include <gtest/gtest.h>
#include "BarnesHutAlgorithm.hpp"
#include "SimulationData.hpp"
#include "sycl.hpp"




TEST(TestTreeCreation, AABB_creation) {
    std::string temp = " ";
    BarnesHutAlgorithm algorithm(1.0/24, 30, 1, temp, 3);
    queue queue;
    std::vector<double> x_positions = {0,0,2};
    std::vector<double> y_positions = {1,0,0};
    std::vector<double> z_positions = {0,2,0};
    SimulationData simulationData;
    simulationData.positions_x = x_positions;
    simulationData.positions_y = y_positions;
    simulationData.positions_z = z_positions;

    algorithm.computeMinMaxValuesAABB(queue, simulationData);

    EXPECT_EQ(algorithm.AABB_EdgeLength, 2);
    EXPECT_EQ(algorithm.min_x, 0);
    EXPECT_EQ(algorithm.min_y, -0.5);
    EXPECT_EQ(algorithm.min_z, 0);
    EXPECT_EQ(algorithm.max_x, 2);
    EXPECT_EQ(algorithm.max_y, 1.5);
    EXPECT_EQ(algorithm.max_z, 2);
}

TEST(TestTreeCreation, split_node_test) {
    std::string temp = " ";
    BarnesHutAlgorithm algorithm(1.0/24, 30, 1, temp, 3);
    queue queue;
    std::vector<double> x_positions = {0,0,2};
    std::vector<double> y_positions = {1,0,0};
    std::vector<double> z_positions = {0,2,0};
    SimulationData simulationData;
    simulationData.positions_x = x_positions;
    simulationData.positions_y = y_positions;
    simulationData.positions_z = z_positions;

    algorithm.computeMinMaxValuesAABB(queue, simulationData);
    // insert root node
    algorithm.edgeLengths.push_back(algorithm.AABB_EdgeLength);
    algorithm.bodyOfNode.push_back(3);
    algorithm.upper_NW.push_back(0);
    algorithm.upper_NE.push_back(0);
    algorithm.upper_SW.push_back(0);
    algorithm.upper_SE.push_back(0);
    algorithm.lower_NW.push_back(0);
    algorithm.lower_NE.push_back(0);
    algorithm.lower_SW.push_back(0);
    algorithm.lower_SE.push_back(0);
    algorithm.min_x_values.push_back(algorithm.min_x);
    algorithm.min_y_values.push_back(algorithm.min_y);
    algorithm.min_z_values.push_back(algorithm.min_z);

    // split the root node into octants
    algorithm.splitNode(0,1);

    // check, that the root node now has 8 children with ids 1 to 8
    EXPECT_EQ(algorithm.upper_NW[0], 1);
    EXPECT_EQ(algorithm.upper_NE[0], 2);
    EXPECT_EQ(algorithm.upper_SW[0], 3);
    EXPECT_EQ(algorithm.upper_SE[0], 4);
    EXPECT_EQ(algorithm.lower_NW[0], 5);
    EXPECT_EQ(algorithm.lower_NE[0], 6);
    EXPECT_EQ(algorithm.lower_SW[0], 7);
    EXPECT_EQ(algorithm.lower_SE[0], 8);

    // check if the min x,y,z values of each new octant have been calculated correctly
    EXPECT_EQ(algorithm.min_x_values[1], 0);
    EXPECT_EQ(algorithm.min_x_values[2], 1);
    EXPECT_EQ(algorithm.min_x_values[3], 0);
    EXPECT_EQ(algorithm.min_x_values[4], 1);
    EXPECT_EQ(algorithm.min_x_values[5], 0);
    EXPECT_EQ(algorithm.min_x_values[6], 1);
    EXPECT_EQ(algorithm.min_x_values[7], 0);
    EXPECT_EQ(algorithm.min_x_values[8], 1);

    EXPECT_EQ(algorithm.min_y_values[1], 0.5);
    EXPECT_EQ(algorithm.min_y_values[2], 0.5);
    EXPECT_EQ(algorithm.min_y_values[3], 0.5);
    EXPECT_EQ(algorithm.min_y_values[4], 0.5);
    EXPECT_EQ(algorithm.min_y_values[5], -0.5);
    EXPECT_EQ(algorithm.min_y_values[6], -0.5);
    EXPECT_EQ(algorithm.min_y_values[7], -0.5);
    EXPECT_EQ(algorithm.min_y_values[8], -0.5);

    EXPECT_EQ(algorithm.min_z_values[1], 0);
    EXPECT_EQ(algorithm.min_z_values[2], 0);
    EXPECT_EQ(algorithm.min_z_values[3], 1);
    EXPECT_EQ(algorithm.min_z_values[4], 1);
    EXPECT_EQ(algorithm.min_z_values[5], 0);
    EXPECT_EQ(algorithm.min_z_values[6], 0);
    EXPECT_EQ(algorithm.min_z_values[7], 1);
    EXPECT_EQ(algorithm.min_z_values[8], 1);

    // check the sizes of all vectors, such that exactly 8 new nodes have been created
    EXPECT_EQ(algorithm.min_x_values.size(), 9);
    EXPECT_EQ(algorithm.min_y_values.size(), 9);
    EXPECT_EQ(algorithm.min_z_values.size(), 9);

    EXPECT_EQ(algorithm.upper_NW.size(), 9);
    EXPECT_EQ(algorithm.upper_NE.size(), 9);
    EXPECT_EQ(algorithm.upper_SW.size(), 9);
    EXPECT_EQ(algorithm.upper_SE.size(), 9);
    EXPECT_EQ(algorithm.lower_NW.size(), 9);
    EXPECT_EQ(algorithm.lower_NE.size(), 9);
    EXPECT_EQ(algorithm.lower_SW.size(), 9);
    EXPECT_EQ(algorithm.lower_SE.size(), 9);

    EXPECT_EQ(algorithm.bodyOfNode.size(), 9);

    // check that none of the nodes contains a body.
    EXPECT_EQ(algorithm.bodyOfNode[0], 3);
    EXPECT_EQ(algorithm.bodyOfNode[1], 3);
    EXPECT_EQ(algorithm.bodyOfNode[2], 3);
    EXPECT_EQ(algorithm.bodyOfNode[3], 3);
    EXPECT_EQ(algorithm.bodyOfNode[4], 3);
    EXPECT_EQ(algorithm.bodyOfNode[5], 3);
    EXPECT_EQ(algorithm.bodyOfNode[6], 3);
    EXPECT_EQ(algorithm.bodyOfNode[7], 3);
    EXPECT_EQ(algorithm.bodyOfNode[8], 3);
}

TEST(TestTreeCreation, getOctantContaingBodyTest) {
    std::string temp = " ";
    BarnesHutAlgorithm algorithm(1.0/24, 30, 1, temp, 3);
    queue queue;
    std::vector<double> x_positions = {0,0,2};
    std::vector<double> y_positions = {1,0,0};
    std::vector<double> z_positions = {0,2,0};

    SimulationData simulationData;

    simulationData.positions_x = x_positions;
    simulationData.positions_y = y_positions;
    simulationData.positions_z = z_positions;

    algorithm.computeMinMaxValuesAABB(queue, simulationData);

    // insert root node
    algorithm.edgeLengths.push_back(algorithm.AABB_EdgeLength);
    algorithm.bodyOfNode.push_back(5);
    algorithm.upper_NW.push_back(0);
    algorithm.upper_NE.push_back(0);
    algorithm.upper_SW.push_back(0);
    algorithm.upper_SE.push_back(0);
    algorithm.lower_NW.push_back(0);
    algorithm.lower_NE.push_back(0);
    algorithm.lower_SW.push_back(0);
    algorithm.lower_SE.push_back(0);
    algorithm.min_x_values.push_back(algorithm.min_x);
    algorithm.min_y_values.push_back(algorithm.min_y);
    algorithm.min_z_values.push_back(algorithm.min_z);

    // split the root node into octants
    algorithm.splitNode(0,1);

    // First Body at (0,1,0) should be in octant upper_NW of the root node.
    EXPECT_EQ(algorithm.getOctantContainingBody(x_positions[0], y_positions[0], z_positions[0], 0), 1);

    // Second Body at (0,0,2) should be in octant lower_SW of the root node.
    EXPECT_EQ(algorithm.getOctantContainingBody(x_positions[1], y_positions[1], z_positions[1], 0), 7);

    // Third Body at (2,0,0) should be in octant lower_NE of the root node
    EXPECT_EQ(algorithm.getOctantContainingBody(x_positions[2], y_positions[2], z_positions[2], 0), 6);
}

TEST(TestTreeCreation, buildOctreeTest) {
    std::string temp = " ";
    BarnesHutAlgorithm algorithm(1.0/24, 30, 1, temp, 3);
    queue queue;
    std::vector<double> x_positions = {0,0,2};
    std::vector<double> y_positions = {1,0,0};
    std::vector<double> z_positions = {0,2,0};

    SimulationData simulationData;

    simulationData.positions_x = x_positions;
    simulationData.positions_y = y_positions;
    simulationData.positions_z = z_positions;

    algorithm.computeMinMaxValuesAABB(queue, simulationData);
    algorithm.buildOctree(queue,x_positions,y_positions,z_positions);

    // check the sizes of all vectors, such that exactly 8 new nodes have been created
    EXPECT_EQ(algorithm.min_x_values.size(), 9);
    EXPECT_EQ(algorithm.min_y_values.size(), 9);
    EXPECT_EQ(algorithm.min_z_values.size(), 9);

    EXPECT_EQ(algorithm.upper_NW.size(), 9);
    EXPECT_EQ(algorithm.upper_NE.size(), 9);
    EXPECT_EQ(algorithm.upper_SW.size(), 9);
    EXPECT_EQ(algorithm.upper_SE.size(), 9);
    EXPECT_EQ(algorithm.lower_NW.size(), 9);
    EXPECT_EQ(algorithm.lower_NE.size(), 9);
    EXPECT_EQ(algorithm.lower_SW.size(), 9);
    EXPECT_EQ(algorithm.lower_SE.size(), 9);

    EXPECT_EQ(algorithm.bodyOfNode.size(), 9);

    // check that the all child nodes contain the correct body.
    EXPECT_EQ(algorithm.bodyOfNode[0], 3);
    EXPECT_EQ(algorithm.bodyOfNode[1], 0); // first body in upper_NW
    EXPECT_EQ(algorithm.bodyOfNode[2], 3);
    EXPECT_EQ(algorithm.bodyOfNode[3], 3);
    EXPECT_EQ(algorithm.bodyOfNode[4], 3);
    EXPECT_EQ(algorithm.bodyOfNode[5], 3);
    EXPECT_EQ(algorithm.bodyOfNode[6], 2); // third body in lower_NE
    EXPECT_EQ(algorithm.bodyOfNode[7], 1); // second body in lower_SW
    EXPECT_EQ(algorithm.bodyOfNode[8], 3);
}


