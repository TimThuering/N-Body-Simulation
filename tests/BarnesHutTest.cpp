 #include <gtest/gtest.h>
//#include "BarnesHutAlgorithm.hpp"
//#include "SimulationData.hpp"
#include <sycl.hpp>
using namespace sycl;



/*
TEST(TestTreeCreation, AABB_creation) {
    std::string temp = " ";
    BarnesHutAlgorithm algorithm(1.0/24, 30, 1, temp, 3);
    queue queue;
    std::vector<double> x_positions_vec = {0,0,2};
    std::vector<double> y_positions_vec = {1,0,0};
    std::vector<double> z_positions_vec = {0,2,0};

    buffer<double> x_positions = x_positions_vec;
    buffer<double> y_positions = y_positions_vec;
    buffer<double> z_positions = z_positions_vec;
    SimulationData simulationData;
    simulationData.positions_x = x_positions_vec;
    simulationData.positions_y = y_positions_vec;
    simulationData.positions_z = z_positions_vec;

    algorithm.computeMinMaxValuesAABB(queue, x_positions, y_positions, z_positions);

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
    std::vector<double> x_positions_vec = {0,0,2};
    std::vector<double> y_positions_vec = {1,0,0};
    std::vector<double> z_positions_vec = {0,2,0};

    buffer<double> x_positions = x_positions_vec;
    buffer<double> y_positions = y_positions_vec;
    buffer<double> z_positions = z_positions_vec;
    SimulationData simulationData;
    simulationData.positions_x = x_positions_vec;
    simulationData.positions_y = y_positions_vec;
    simulationData.positions_z = z_positions_vec;

    algorithm.computeMinMaxValuesAABB(queue, x_positions, y_positions, z_positions);
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
    std::vector<double> x_positions_vec = {0,0,2};
    std::vector<double> y_positions_vec = {1,0,0};
    std::vector<double> z_positions_vec = {0,2,0};

    buffer<double> x_positions = x_positions_vec;
    buffer<double> y_positions = y_positions_vec;
    buffer<double> z_positions = z_positions_vec;
    SimulationData simulationData;
    simulationData.positions_x = x_positions_vec;
    simulationData.positions_y = y_positions_vec;
    simulationData.positions_z = z_positions_vec;

    algorithm.computeMinMaxValuesAABB(queue, x_positions, y_positions, z_positions);

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

    host_accessor<double> POS_X(x_positions);
    host_accessor<double> POS_Y(y_positions);
    host_accessor<double> POS_Z(z_positions);

    // First Body at (0,1,0) should be in octant upper_NW of the root node.
    EXPECT_EQ(algorithm.getOctantContainingBody(POS_X[0], POS_Y[0], POS_Z[0], 0), 1);

    // Second Body at (0,0,2) should be in octant lower_SW of the root node.
    EXPECT_EQ(algorithm.getOctantContainingBody(POS_X[1], POS_Y[1], POS_Z[1], 0), 7);

    // Third Body at (2,0,0) should be in octant lower_NE of the root node
    EXPECT_EQ(algorithm.getOctantContainingBody(POS_X[2], POS_Y[2], POS_Z[2], 0), 6);
}

TEST(TestTreeCreation, buildOctreeTest) {
    std::string temp = " ";
    BarnesHutAlgorithm algorithm(1.0/24, 30, 1, temp, 3);
    queue queue;
    std::vector<double> x_positions_vec = {0,0,2};
    std::vector<double> y_positions_vec = {1,0,0};
    std::vector<double> z_positions_vec = {0,2,0};
    std::vector<double> masses_vec = {10,10,10};

    buffer<double> x_positions = x_positions_vec;
    buffer<double> y_positions = y_positions_vec;
    buffer<double> z_positions = z_positions_vec;
    buffer<double> masses = masses_vec;

    SimulationData simulationData;

    simulationData.positions_x = x_positions_vec;
    simulationData.positions_y = y_positions_vec;
    simulationData.positions_z = z_positions_vec;
    simulationData.mass = masses_vec;

    algorithm.computeMinMaxValuesAABB(queue, x_positions, y_positions, z_positions);
    algorithm.buildOctree(queue,x_positions,y_positions,z_positions, masses);

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

TEST(TestTreeCreation, fenceTest) {

    queue queue;

    int n = 500;


    std::vector<int> Test_vec(n + 1, -1);
    buffer<int> Test = Test_vec;

    std::vector<int> Test_vec2(n + 1, -1);
    buffer<int> Test2 = Test_vec2;


    std::vector<std::size_t> nextFreeID_vec(1, 1);
    buffer<std::size_t> nextFreeID = nextFreeID_vec;
//
    std::vector<int> lock_vec(1, 0);
    buffer<int> lock = lock_vec;


    queue.submit([&](handler &h) {


//
//
        accessor<std::size_t> NEXT_FREE_ID(nextFreeID, h);
        accessor<int> LOCK(lock, h);
        accessor<int> TEST_ACC(Test, h);
        accessor<int> TEST_ACC2(Test2, h);


        h.parallel_for(nd_range<1>(range<1>(n), range<1>(n)), [=](auto &i) {
//        h.parallel_for(n, [=](auto &i) {




            int index = 0;


            bool success = false;
            int k = 0;
//
            while (!success) {


                int exp = 0;
                atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                        access::address_space::global_space> lockAccessor(LOCK[index]);

                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                        access::address_space::global_space> nextFreeIDAccessor(NEXT_FREE_ID[0]);


//

                if (lockAccessor.compare_exchange_strong(exp, 1, memory_order::acq_rel,
                                                         memory_scope::device)) {
                    std::size_t nextFreeID;
                    nextFreeID = nextFreeIDAccessor.fetch_add(1);
//                    TEST_ACC[nextFreeID] = 50000;
//
//                    i.barrier();
//                                       sycl::atomic_fence(memory_order::acq_rel, memory_scope::device);
                    lockAccessor.fetch_sub(1, memory_order::acq_rel, memory_scope::device);

                    success = true;


                }

                k++;

            }


        });
    }).wait();

    host_accessor acc(Test2);
    host_accessor acc2(nextFreeID);

//    for (int i = 0; i < n + 1; ++i) {
//        std::cout << acc[i] << std::endl;
//    }
    std::cout << acc2[0];

}
 */

TEST(TestTreeCreation, testMemFence) {

    queue queue;

    int size = 64000;

    std::vector<long> testVec(size + 1, -1);

    std::vector<long> resVec(size + 1);

    std::vector<int> lockVec(1, 0);

    std::vector<int> counterVec(1, 1);

    buffer<long> testBuff = testVec;

    buffer<long> resBuff = resVec;

    buffer<int> lockBuff = lockVec;

    buffer<int> counterBuff = counterVec;


    queue.submit([&](handler &h) {

        accessor data(testBuff, h);

        accessor lock(lockBuff, h);

        accessor res(resBuff, h);

        accessor counter(counterBuff, h);







         h.parallel_for(size, [=](auto &i) {

//        h.parallel_for(nd_range<1>(range<1>(size), range<1>(size)), [=](auto &nd_item) {


            int exp = 0;

            atomic_ref<int, memory_order::acq_rel, memory_scope::device,

                    access::address_space::global_space> atomicNodeIsLockedAccessor(

                    lock[0]);


            atomic_ref<int, memory_order::acq_rel, memory_scope::device,

                    access::address_space::global_space> atomicCounter(

                    counter[0]);


            bool operationDone = false;


            while (!operationDone) {




                if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1, memory_order::acq_rel,
                                                                       memory_scope::device)) {

                    int index = atomicCounter.fetch_add(1, memory_order::acq_rel, memory_scope::device);

//                    data[index] = nd_item.get_global_id();
                    data[index] = i;

//                    nd_item.mem_fence(access::fence_space::global_and_local);
                    atomic_fence(memory_order::acq_rel,memory_scope::device);

//                    nd_item.mem_fence(access::fence_space::global_and_local);


                    operationDone = true;

                    atomicNodeIsLockedAccessor.fetch_sub(1, memory_order::acq_rel, memory_scope::device);
//                    res[nd_item.get_global_id()] = data[index - 1];
                                        res[i] = data[index - 1];




                }

            }


        });

    }).wait();


    host_accessor test(resBuff);

    for (int i = 0; i <= size; ++i) {

        if (test[i] == -1) {
            std::cout << i << " :  " << test[i] << std::endl;

        }


    }

}


 TEST(TestTreeCreation, testrsqrt) {

     queue queue;

     int size = 64000;

     std::vector<double> testVec(size + 1, -1);



     buffer<double> testBuff = testVec;



     queue.submit([&](handler &h) {

         accessor data(testBuff, h);









         h.single_task( [=]() {
             double x1 = 500000000.123546;
             double x2 = 500000000.123546;
             for (int i = 0; i < 10; ++i) {

                 x1 = 1 / sycl::sqrt(x1);
                 x2 = sycl::rsqrt(x2);


             }

             data[0] = x1;
             data[1] = x2;





         });

     }).wait();


     host_accessor test(testBuff);

     double one_div_sqrt_cpp = 500000000.123546;


     double rsqrt_sycl = test[0];
     double one_div_sqrt_sycl = test[1];
     for (int i = 0; i < 10; ++i) {
         one_div_sqrt_cpp= 1.0 / std::sqrt(one_div_sqrt_cpp);
     }

     std::cout.precision(1000);
     std::cout << test[0] << std::endl;
     std::cout << test[1] << std::endl;
     std::cout << one_div_sqrt_cpp << std::endl;

     std::cout << "";



 }

