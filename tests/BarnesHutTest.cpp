#include <gtest/gtest.h>
#include "BarnesHutAlgorithm.hpp"
#include "ParallelOctreeTopDownSubtrees.hpp"
#include "BarnesHutOctree.hpp"
#include "SimulationData.hpp"
#include <sycl.hpp>
#include "Configuration.hpp"

using namespace sycl;

TEST(TestTreeCreation, AABB_creation) {
    ParallelOctreeTopDownSubtrees octree;
    queue queue;
    std::vector<double> x_positions_vec = {0, 0, 2};
    std::vector<double> y_positions_vec = {1, 0, 0};
    std::vector<double> z_positions_vec = {0, 2, 0};

    configuration::numberOfBodies = 3;

    buffer<double> x_positions(x_positions_vec.data(), x_positions_vec.size());
    buffer<double> y_positions(y_positions_vec.data(), y_positions_vec.size());
    buffer<double> z_positions(z_positions_vec.data(), z_positions_vec.size());

    octree.computeMinMaxValuesAABB(queue, x_positions, y_positions, z_positions);

    EXPECT_EQ(octree.AABB_EdgeLength, 2);
    EXPECT_EQ(octree.min_x, 0);
    EXPECT_EQ(octree.min_y, -0.5);
    EXPECT_EQ(octree.min_z, 0);
    EXPECT_EQ(octree.max_x, 2);
    EXPECT_EQ(octree.max_y, 1.5);
    EXPECT_EQ(octree.max_z, 2);
}

TEST(TestTreeCreation, buildOctreeTest) {
    configuration::numberOfBodies = 3;
    configuration::initializeConfigValues(3, 16, 16);
    configuration::barnes_hut_algorithm::octreeWorkItemCount = 3;

    ParallelOctreeTopDownSynchronized octree;
    TimeMeasurement timer;
    queue queue;
    std::vector<double> x_positions_vec = {0, 0, 2};
    std::vector<double> y_positions_vec = {1, 0, 0};
    std::vector<double> z_positions_vec = {0, 2, 0};
    std::vector<double> masses_vec = {10, 10, 10};

    buffer<double> x_positions = x_positions_vec;
    buffer<double> y_positions = y_positions_vec;
    buffer<double> z_positions = z_positions_vec;
    buffer<double> masses = masses_vec;

    SimulationData simulationData;

    simulationData.positions_x = x_positions_vec;
    simulationData.positions_y = y_positions_vec;
    simulationData.positions_z = z_positions_vec;
    simulationData.mass = masses_vec;

    octree.buildOctree(queue, x_positions, y_positions, z_positions, masses, timer);

    // check that the all child nodes contain the correct body.
    host_accessor<d_type::int_t> BODY_OF_NODE(octree.bodyOfNode);
    EXPECT_EQ(BODY_OF_NODE[0], 3);
    EXPECT_EQ(BODY_OF_NODE[1], 0); // first body in upper_NW
    EXPECT_EQ(BODY_OF_NODE[2], 3);
    EXPECT_EQ(BODY_OF_NODE[3], 3);
    EXPECT_EQ(BODY_OF_NODE[4], 3);
    EXPECT_EQ(BODY_OF_NODE[5], 3);
    EXPECT_EQ(BODY_OF_NODE[6], 2); // third body in lower_NE
    EXPECT_EQ(BODY_OF_NODE[7], 1); // second body in lower_SW
    EXPECT_EQ(BODY_OF_NODE[8], 3);
}

TEST(TestTreeCreation, buildOctreeSubtreesTest) {
    configuration::numberOfBodies = 3;
    configuration::initializeConfigValues(3, 3, 3);
    configuration::barnes_hut_algorithm::sortBodies = true;
    configuration::barnes_hut_algorithm::octreeWorkItemCount = 3;
    configuration::barnes_hut_algorithm::octreeTopWorkItemCount = 3;
    configuration::barnes_hut_algorithm::maxBuildLevel = 1;

    ParallelOctreeTopDownSubtrees octree;
    TimeMeasurement timer;
    queue queue;
    std::vector<double> x_positions_vec = {0, 0, 2};
    std::vector<double> y_positions_vec = {1, 0, 0};
    std::vector<double> z_positions_vec = {0, 2, 0};
    std::vector<double> masses_vec = {10, 10, 10};

    buffer<double> x_positions = x_positions_vec;
    buffer<double> y_positions = y_positions_vec;
    buffer<double> z_positions = z_positions_vec;
    buffer<double> masses = masses_vec;

    SimulationData simulationData;

    simulationData.positions_x = x_positions_vec;
    simulationData.positions_y = y_positions_vec;
    simulationData.positions_z = z_positions_vec;
    simulationData.mass = masses_vec;

    octree.buildOctree(queue, x_positions, y_positions, z_positions, masses, timer);

    // check that the all child nodes contain the correct body.
    host_accessor<d_type::int_t> BODY_OF_NODE(octree.bodyOfNode);

    EXPECT_EQ(BODY_OF_NODE[0], 3);
    EXPECT_EQ(BODY_OF_NODE[1], 0); // first body in upper_NW
    EXPECT_EQ(BODY_OF_NODE[2], 3);
    EXPECT_EQ(BODY_OF_NODE[3], 3);
    EXPECT_EQ(BODY_OF_NODE[4], 3);
    EXPECT_EQ(BODY_OF_NODE[5], 3);
    EXPECT_EQ(BODY_OF_NODE[6], 2); // third body in lower_NE
    EXPECT_EQ(BODY_OF_NODE[7], 1); // second body in lower_SW
    EXPECT_EQ(BODY_OF_NODE[8], 3);

    // test compute center of Mass: root node has to store the sum of all masses
    host_accessor<double> SUM_MASSES(octree.sumOfMasses);
    EXPECT_FLOAT_EQ(SUM_MASSES[0],30.0);

    // test if bodies have been sorted correctly
    host_accessor<d_type::int_t> SORTED_BODIES(octree.sortedBodiesInOrder);
    EXPECT_EQ(SORTED_BODIES[0], 1);
    EXPECT_EQ(SORTED_BODIES[1], 2);
    EXPECT_EQ(SORTED_BODIES[2], 0);
}

TEST(TestTreeCreation, prepareSubtreesTest) {
    queue queue;
    configuration::numberOfBodies = 10;
    configuration::initializeConfigValues(10, 16, 16);
    ParallelOctreeTopDownSubtrees octree;
    d_type::int_t nodeCount = 9;

    // 4 bodies with subtree root node 4, 2 bodies with node 4,5 respectively and one body with root node 7-
    // One body has reached its final position,so it has the root node 0 as subtree.
    octree.subtreeOfBody_vec = {1, 1, 1, 1, 0, 4, 4, 5, 5, 7};

    std::vector<d_type::int_t> bodyCountSubtree_vec(nodeCount, 0);
    buffer<d_type::int_t> bodyCountSubtree(bodyCountSubtree_vec.data(), bodyCountSubtree_vec.size());

    std::vector<d_type::int_t> subtrees_vec(nodeCount);
    buffer<d_type::int_t> subtrees(subtrees_vec.data(), subtrees_vec.size());

    octree.prepareSubtrees(queue, bodyCountSubtree, subtrees, nodeCount);

    host_accessor<d_type::int_t> SUBTREES(subtrees);
    EXPECT_EQ(SUBTREES[0], 1);
    EXPECT_EQ(SUBTREES[1], 4);
    EXPECT_EQ(SUBTREES[2], 5);
    EXPECT_EQ(SUBTREES[3], 7);

    host_accessor<d_type::int_t> BODY_COUNT_SUBTREE(bodyCountSubtree);
    EXPECT_EQ(BODY_COUNT_SUBTREE[0], 1);
    EXPECT_EQ(BODY_COUNT_SUBTREE[1], 4);
    EXPECT_EQ(BODY_COUNT_SUBTREE[2], 0);
    EXPECT_EQ(BODY_COUNT_SUBTREE[3], 0);
    EXPECT_EQ(BODY_COUNT_SUBTREE[4], 2);
    EXPECT_EQ(BODY_COUNT_SUBTREE[5], 2);
    EXPECT_EQ(BODY_COUNT_SUBTREE[6], 0);
    EXPECT_EQ(BODY_COUNT_SUBTREE[7], 1);
    EXPECT_EQ(BODY_COUNT_SUBTREE[8], 0);
    EXPECT_EQ(BODY_COUNT_SUBTREE[9], 0);

    host_accessor<d_type::int_t> SUBTREE_COUNT(octree.subtreeCount);
    EXPECT_EQ(SUBTREE_COUNT[0], 4); // 4 Subtrees with root nodes 1,4,5,7
}

TEST(TestTreeCreation, TestSortBodiesForSubtrees) {
    queue queue;
    configuration::numberOfBodies = 10;
    configuration::initializeConfigValues(10, 16, 16);
    ParallelOctreeTopDownSubtrees octree;
    d_type::int_t nodeCount = 9;

    // 4 bodies with subtree root node 4, 2 bodies with node 4,5 respectively and one body with root node 7-
    // One body has reached its final position,so it has the root node 0 as subtree.
    octree.subtreeOfBody_vec = {1, 1, 1, 1, 0, 4, 4, 5, 5, 7};

    std::vector<d_type::int_t> bodyCountSubtree_vec(nodeCount, 0);
    buffer<d_type::int_t> bodyCountSubtree(bodyCountSubtree_vec.data(), bodyCountSubtree_vec.size());

    std::vector<d_type::int_t> subtrees_vec(nodeCount);
    buffer<d_type::int_t> subtrees(subtrees_vec.data(), subtrees_vec.size());

    octree.prepareSubtrees(queue, bodyCountSubtree, subtrees, nodeCount);

    {
        host_accessor<d_type::int_t> SUBTREE_COUNT(octree.subtreeCount);
        octree.numberOfSubtrees = SUBTREE_COUNT[0];
    }

    // stores for each subtree where the indices of the bodies belonging to this subtree start
    std::vector<d_type::int_t> bodiesOfSubtreeStartIndex_vec(octree.numberOfSubtrees, 0);
    buffer<d_type::int_t> bodiesOfSubtreeStartIndex(bodiesOfSubtreeStartIndex_vec.data(),
                                                    bodiesOfSubtreeStartIndex_vec.size());

    octree.sortBodiesForSubtrees(queue, bodyCountSubtree, subtrees, bodiesOfSubtreeStartIndex);

    host_accessor<d_type::int_t> SUBTREE_COUNT(octree.subtreeCount);
    EXPECT_EQ(SUBTREE_COUNT[0], 4); // 4 Subtrees with root nodes 1,4,5,7

    host_accessor<d_type::int_t> SORTED_BODIES(octree.sortedBodies);
    EXPECT_EQ(SORTED_BODIES[0],0);
    EXPECT_EQ(SORTED_BODIES[1],1);
    EXPECT_EQ(SORTED_BODIES[2],2);
    EXPECT_EQ(SORTED_BODIES[3],3);
    EXPECT_EQ(SORTED_BODIES[4],5);
    EXPECT_EQ(SORTED_BODIES[5],6);
    EXPECT_EQ(SORTED_BODIES[6],7);
    EXPECT_EQ(SORTED_BODIES[7],8);
    EXPECT_EQ(SORTED_BODIES[8],9);

    host_accessor<d_type::int_t> START_INDEX(bodiesOfSubtreeStartIndex);
    EXPECT_EQ(START_INDEX[0],0); // 4 bodies
    EXPECT_EQ(START_INDEX[1],4); // 2 bodies
    EXPECT_EQ(START_INDEX[2],6); // 2 bodies
    EXPECT_EQ(START_INDEX[3],8); // 1 body
}