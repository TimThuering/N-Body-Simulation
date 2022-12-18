#include "BarnesHutAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>
#include "list"
#include "stack"
#include <chrono>

using namespace sycl;

BarnesHutAlgorithm::BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth,
                                       std::string &outputDirectory, std::size_t numberOfBodies)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory, numberOfBodies),
          upper_NW_vec(16 * numberOfBodies),
          upper_NE_vec(16 * numberOfBodies),
          upper_SW_vec(16 * numberOfBodies),
          upper_SE_vec(16 * numberOfBodies),
          lower_NW_vec(16 * numberOfBodies),
          lower_NE_vec(16 * numberOfBodies),
          lower_SW_vec(16 * numberOfBodies),
          lower_SE_vec(16 * numberOfBodies),
          edgeLengths_vec(16 * numberOfBodies),
          min_x_values_vec(16 * numberOfBodies),
          min_y_values_vec(16 * numberOfBodies),
          min_z_values_vec(16 * numberOfBodies),
          bodyOfNode_vec(16 * numberOfBodies, numberOfBodies),
          sumMasses_vec(16 * numberOfBodies),
          centerOfMass_x_vec(16 * numberOfBodies),
          centerOfMass_y_vec(16 * numberOfBodies),
          centerOfMass_z_vec(16 * numberOfBodies) {
    this->description = "Barnes-Hut Algorithm";
}


void BarnesHutAlgorithm::startSimulation(const SimulationData &simulationData) {
    // Vectors and their corresponding buffers for the acceleration values of each body induced by the gravitational
    // force from all other bodies in the dataset. The index corresponds the body id.
    std::vector<double> acc_x(numberOfBodies);
    std::vector<double> acc_y(numberOfBodies);
    std::vector<double> acc_z(numberOfBodies);

    buffer<double> acceleration_x = acc_x;
    buffer<double> acceleration_y = acc_y;
    buffer<double> acceleration_z = acc_z;

    // set initial values of position and velocity to the values read directly form the dataset
    positions_x[0] = simulationData.positions_x;
    positions_y[0] = simulationData.positions_y;
    positions_z[0] = simulationData.positions_z;

    velocities_x[0] = simulationData.velocities_x;
    velocities_y[0] = simulationData.velocities_y;
    velocities_z[0] = simulationData.velocities_z;

    // adjust the initial velocities of all bodies
    adjustVelocities(simulationData);

    // buffers for intermediate position and velocity values;
    std::vector<double> intermediatePosition_x_vec = simulationData.positions_x;
    std::vector<double> intermediatePosition_y_vec = simulationData.positions_y;
    std::vector<double> intermediatePosition_z_vec = simulationData.positions_z;

    std::vector<double> intermediateVelocity_x_vec = simulationData.velocities_x;
    std::vector<double> intermediateVelocity_y_vec = simulationData.velocities_y;
    std::vector<double> intermediateVelocity_z_vec = simulationData.velocities_z;

    buffer<double> intermediatePosition_x = intermediatePosition_x_vec;
    buffer<double> intermediatePosition_y = intermediatePosition_y_vec;
    buffer<double> intermediatePosition_z = intermediatePosition_z_vec;

    buffer<double> intermediateVelocity_x = intermediateVelocity_x_vec;
    buffer<double> intermediateVelocity_y = intermediateVelocity_y_vec;
    buffer<double> intermediateVelocity_z = intermediateVelocity_z_vec;

    // SYCL queue for computation tasks
    queue queue;

    // vector containing all the masses of the bodies
    std::vector<double> masses_vec = simulationData.mass;
    buffer masses = simulationData.mass;


    // current time of the simulation
    double time = 0.0;
    double timeSinceLastVisualization = 0.0; // used to determine the time steps that should be visualized

    // current visualization step of the simulation
    std::size_t currentStep = 0;

    // start of the simulation:
    // computations for initial state: all values get stored for the output

    computeMinMaxValuesAABB(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z);
    buildOctreeParallel(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z, masses);


    // compute initial accelerations
    computeAccelerations(queue, masses, intermediatePosition_x, intermediatePosition_y,
                         intermediatePosition_z,
                         acceleration_x, acceleration_y, acceleration_z);


    // compute energy of the initial step.
    computeEnergy(queue, masses, currentStep, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z,
                  intermediateVelocity_x, intermediateVelocity_y, intermediateVelocity_z);


    // store norm of all accelerations
    storeAccelerations(currentStep, acceleration_x, acceleration_y, acceleration_z);
    std::cout << "Step " << currentStep << std::endl;

    // continue with next simulation step
    time += dt;
    timeSinceLastVisualization += dt;
    currentStep += 1;

    // vectors for intermediate values during the time integration
    buffer<double> vx_k1_2(numberOfBodies);
    buffer<double> vy_k1_2(numberOfBodies);
    buffer<double> vz_k1_2(numberOfBodies);


    // simulation of the next steps:
    while (time < t_end) {

        // determine if current step should be visualized.
        bool visualizeCurrentStep = (std::abs(timeSinceLastVisualization - visualizationStepWidth) < 0.000001);

        double delta_t = dt;

        // leapfrog integration part 1: update the position
        queue.submit([&](handler &h) {

            accessor<double> VX_k1_2(vx_k1_2, h);
            accessor<double> VY_k1_2(vy_k1_2, h);
            accessor<double> VZ_k1_2(vz_k1_2, h);

            accessor<double> INTERMEDIATE_V_X(intermediateVelocity_x, h);
            accessor<double> INTERMEDIATE_V_Y(intermediateVelocity_y, h);
            accessor<double> INTERMEDIATE_V_Z(intermediateVelocity_z, h);

            accessor<double> INTERMEDIATE_P_X(intermediatePosition_x, h);
            accessor<double> INTERMEDIATE_P_Y(intermediatePosition_y, h);
            accessor<double> INTERMEDIATE_P_Z(intermediatePosition_z, h);

            accessor<double> ACC_X(acceleration_x, h);
            accessor<double> ACC_Y(acceleration_y, h);
            accessor<double> ACC_Z(acceleration_z, h);

            h.parallel_for(numberOfBodies, [=](auto &i) {

                VX_k1_2[i] = INTERMEDIATE_V_X[i] + ACC_X[i] * (delta_t / 2.0);
                VY_k1_2[i] = INTERMEDIATE_V_Y[i] + ACC_Y[i] * (delta_t / 2.0);
                VZ_k1_2[i] = INTERMEDIATE_V_Z[i] + ACC_Z[i] * (delta_t / 2.0);

                // get new position
                INTERMEDIATE_P_X[i] = INTERMEDIATE_P_X[i] + VX_k1_2[i] * delta_t;
                INTERMEDIATE_P_Y[i] = INTERMEDIATE_P_Y[i] + VY_k1_2[i] * delta_t;
                INTERMEDIATE_P_Z[i] = INTERMEDIATE_P_Z[i] + VZ_k1_2[i] * delta_t;
            });
        }).wait();

        // store the current body positions for visualization if the current step should be visualized
        if (visualizeCurrentStep) {
            std::cout << "Step " << currentStep << std::endl;

            host_accessor<double> INTERMEDIATE_P_X(intermediatePosition_x);
            host_accessor<double> INTERMEDIATE_P_Y(intermediatePosition_y);
            host_accessor<double> INTERMEDIATE_P_Z(intermediatePosition_z);

            for (std::size_t i = 0; i < numberOfBodies; ++i) {
                positions_x[currentStep].push_back(INTERMEDIATE_P_X[i]);
                positions_y[currentStep].push_back(INTERMEDIATE_P_Y[i]);
                positions_z[currentStep].push_back(INTERMEDIATE_P_Z[i]);
            }
        }

//        {
//            host_accessor<double> SUM_MASSES(sumOfMasses);
//            std::cout << "----------------------- " << SUM_MASSES[0] << std::endl;
//        }

        auto begin = std::chrono::steady_clock::now();
        resetOctree();
        computeMinMaxValuesAABB(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z);
        buildOctreeParallel(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z,
                            masses);

        computeAccelerations(queue, masses, intermediatePosition_x, intermediatePosition_y,
                             intermediatePosition_z,
                             acceleration_x, acceleration_y, acceleration_z);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Time of step:  " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << std::endl;


        // leapfrog integration part 2: update velocities based on the newly computed acceleration
        queue.submit([&](handler &h) {

            accessor<double> VX_k1_2(vx_k1_2, h);
            accessor<double> VY_k1_2(vy_k1_2, h);
            accessor<double> VZ_k1_2(vz_k1_2, h);

            accessor<double> INTERMEDIATE_V_X(intermediateVelocity_x, h);
            accessor<double> INTERMEDIATE_V_Y(intermediateVelocity_y, h);
            accessor<double> INTERMEDIATE_V_Z(intermediateVelocity_z, h);

            accessor<double> ACC_X(acceleration_x, h);
            accessor<double> ACC_Y(acceleration_y, h);
            accessor<double> ACC_Z(acceleration_z, h);

            h.parallel_for(numberOfBodies, [=](auto &i) {
                INTERMEDIATE_V_X[i] = VX_k1_2[i] + ACC_X[i] * (delta_t / 2.0);
                INTERMEDIATE_V_Y[i] = VY_k1_2[i] + ACC_Y[i] * (delta_t / 2.0);
                INTERMEDIATE_V_Z[i] = VZ_k1_2[i] + ACC_Z[i] * (delta_t / 2.0);
            });
        }).wait();

        if (visualizeCurrentStep) {
            // store norm of all accelerations for the output
            storeAccelerations(currentStep, acceleration_x, acceleration_y, acceleration_z);

            {
                host_accessor<double> INTERMEDIATE_V_X(intermediateVelocity_x);
                host_accessor<double> INTERMEDIATE_V_Y(intermediateVelocity_y);
                host_accessor<double> INTERMEDIATE_V_Z(intermediateVelocity_z);

                // store current velocities for visualization
                for (std::size_t i = 0; i < numberOfBodies; ++i) {
                    velocities_x[currentStep].push_back(INTERMEDIATE_V_X[i]);
                    velocities_y[currentStep].push_back(INTERMEDIATE_V_Y[i]);
                    velocities_z[currentStep].push_back(INTERMEDIATE_V_Z[i]);
                }
            }

            // compute and store energy values of the system in the current time step
            computeEnergy(queue, masses, currentStep, intermediatePosition_x, intermediatePosition_y,
                          intermediatePosition_z, intermediateVelocity_x, intermediateVelocity_y,
                          intermediateVelocity_z);

            // reset time for next visualization time step
            currentStep += 1;
            timeSinceLastVisualization = 0.0;
        }



        // continue with next simulation step
        time += dt;
        timeSinceLastVisualization += dt;

    }


}

void BarnesHutAlgorithm::resetOctree() {
    min_x = std::numeric_limits<double>::infinity();
    min_y = std::numeric_limits<double>::infinity();
    min_z = std::numeric_limits<double>::infinity();
    max_x = std::numeric_limits<double>::lowest();
    max_y = std::numeric_limits<double>::lowest();
    max_z = std::numeric_limits<double>::lowest();

    upper_NW_vec.clear();
    upper_NE_vec.clear();
    upper_SW_vec.clear();
    upper_SE_vec.clear();

    lower_NW_vec.clear();
    lower_NE_vec.clear();
    lower_SW_vec.clear();
    lower_SE_vec.clear();

    edgeLengths_vec.clear();

    min_x_values_vec.clear();
    min_y_values_vec.clear();
    min_z_values_vec.clear();

    bodyOfNode_vec.clear();

    sumMasses_vec.clear();

    centerOfMass_x_vec.clear();
    centerOfMass_y_vec.clear();
    centerOfMass_z_vec.clear();
}

void
BarnesHutAlgorithm::buildOctree(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                                buffer<double> &current_positions_z, buffer<double> &masses) {

    host_accessor<double> POS_X(current_positions_x);
    host_accessor<double> POS_Y(current_positions_y);
    host_accessor<double> POS_Z(current_positions_z);

    host_accessor<double> MASSES(masses);

    host_accessor<std::size_t> UPPER_NW(upper_NW);
    host_accessor<std::size_t> UPPER_NE(upper_NE);
    host_accessor<std::size_t> UPPER_SW(upper_SW);
    host_accessor<std::size_t> UPPER_SE(upper_SE);

    host_accessor<std::size_t> LOWER_NW(lower_NW);
    host_accessor<std::size_t> LOWER_NE(lower_NE);
    host_accessor<std::size_t> LOWER_SW(lower_SW);
    host_accessor<std::size_t> LOWER_SE(lower_SE);

    host_accessor<double> EDGE_LENGTHS(edgeLengths);

    host_accessor<double> MIN_X(min_x_values);
    host_accessor<double> MIN_Y(min_y_values);
    host_accessor<double> MIN_Z(min_z_values);

    host_accessor<std::size_t> BODY_OF_NODE(bodyOfNode);

    host_accessor<double> SUM_MASSES(sumOfMasses);

    host_accessor<double> CENTER_OF_MASS_X(massCenters_x);
    host_accessor<double> CENTER_OF_MASS_Y(massCenters_y);
    host_accessor<double> CENTER_OF_MASS_Z(massCenters_z);

    std::size_t nextFreeNodeID = 0;
    std::size_t currentBody = 0;

    maxTreeDepth = 0;

    bool print = false;
    for (int n = 0; n < 16 * numberOfBodies; n++) {
        if ((n - 1) % 8 == 0) {
            print = true;
        }
        if (BODY_OF_NODE[n] != 178) {
            if (print) {
                // std::cout << "-----------------------------" << std::endl;
                print = false;
            }
            //std::cout << n << ":  " << BODY_OF_NODE[n] << std::endl;
        }


    }


    // root node 0: the AABB of all bodies
    EDGE_LENGTHS[0] = AABB_EdgeLength;
    MIN_X[0] = min_x;
    MIN_Y[0] = min_y;
    MIN_Z[0] = min_z;

    // root has no children yet.
    UPPER_NW[0] = 0;
    UPPER_NE[0] = 0;
    UPPER_SW[0] = 0;
    UPPER_SE[0] = 0;
    LOWER_NW[0] = 0;
    LOWER_NE[0] = 0;
    LOWER_SW[0] = 0;
    LOWER_SE[0] = 0;


    BODY_OF_NODE[0] = currentBody; // insert the first body into the root.

    // initialize sum Masses and centerOfMass vectors
    SUM_MASSES[0] = MASSES[currentBody];

    CENTER_OF_MASS_X[0] = POS_X[currentBody] * MASSES[currentBody];
    CENTER_OF_MASS_Y[0] = POS_Y[currentBody] * MASSES[currentBody];
    CENTER_OF_MASS_Z[0] = POS_Z[currentBody] * MASSES[currentBody];

    nextFreeNodeID += 1;


    for (std::size_t i = 1; i < numberOfBodies; ++i) {
        std::size_t currentDepth = 0;
        std::size_t currentNode = 0;
        bool nodeInserted = false;
        while (!nodeInserted) {
            if (UPPER_NW[currentNode] == 0) {
                // the current node is a leaf node
                if (BODY_OF_NODE[currentNode] == numberOfBodies) {
                    // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body
                    BODY_OF_NODE[currentNode] = i;

                    // update sum masses and center of mass
                    SUM_MASSES[currentNode] += MASSES[i];
                    CENTER_OF_MASS_X[currentNode] += POS_X[i] * MASSES[i];
                    CENTER_OF_MASS_Y[currentNode] += POS_Y[i] * MASSES[i];
                    CENTER_OF_MASS_Z[currentNode] += POS_Z[i] * MASSES[i];

                    if (currentDepth > maxTreeDepth) {
                        maxTreeDepth = currentDepth;
                    }
                    nodeInserted = true;
                } else {
                    // the leaf node already contains a body --> split the node and insert old body
                    std::size_t bodyIDinNode = BODY_OF_NODE[currentNode];
                    splitNode(currentNode, nextFreeNodeID, UPPER_NW, UPPER_NE, UPPER_SW, UPPER_SE, LOWER_NW, LOWER_NE,
                              LOWER_SW, LOWER_SE, EDGE_LENGTHS, MIN_X, MIN_Y, MIN_Z, BODY_OF_NODE, SUM_MASSES,
                              CENTER_OF_MASS_X, CENTER_OF_MASS_Y, CENTER_OF_MASS_Z);
                    nextFreeNodeID += 8;

                    std::size_t octantID = getOctantContainingBody(POS_X[bodyIDinNode],
                                                                   POS_Y[bodyIDinNode],
                                                                   POS_Z[bodyIDinNode], currentNode, UPPER_NW, UPPER_NE,
                                                                   UPPER_SW, UPPER_SE, LOWER_NW, LOWER_NE, LOWER_SW,
                                                                   LOWER_SE, EDGE_LENGTHS, MIN_X, MIN_Y, MIN_Z);
                    // insert the old body into the new octant it belongs to and remove it from the parent node
                    BODY_OF_NODE[octantID] = bodyIDinNode;
                    BODY_OF_NODE[currentNode] = numberOfBodies;

                    SUM_MASSES[octantID] += MASSES[bodyIDinNode];
                    CENTER_OF_MASS_X[octantID] += POS_X[bodyIDinNode] * MASSES[bodyIDinNode];
                    CENTER_OF_MASS_Y[octantID] += POS_Y[bodyIDinNode] * MASSES[bodyIDinNode];
                    CENTER_OF_MASS_Z[octantID] += POS_Z[bodyIDinNode] * MASSES[bodyIDinNode];

                }
            } else {
                // the current node is not a leaf node, i.e. it has 8 children
                // --> determine the octant, the body has to be inserted and set this octant as current node.

                // update sum masses and center of mass of this node, since the current body will be inserted in one of the children
                SUM_MASSES[currentNode] += MASSES[i];
                CENTER_OF_MASS_X[currentNode] += POS_X[i] * MASSES[i];
                CENTER_OF_MASS_Y[currentNode] += POS_Y[i] * MASSES[i];
                CENTER_OF_MASS_Z[currentNode] += POS_Z[i] * MASSES[i];

                std::size_t octantID = getOctantContainingBody(POS_X[i], POS_Y[i],
                                                               POS_Z[i], currentNode, UPPER_NW, UPPER_NE, UPPER_SW,
                                                               UPPER_SE, LOWER_NW, LOWER_NE, LOWER_SW, LOWER_SE,
                                                               EDGE_LENGTHS, MIN_X, MIN_Y, MIN_Z);

                currentNode = octantID;
                currentDepth += 1;

            }

        }
    }

    std::cout << SUM_MASSES[0] << std::endl;


}

void BarnesHutAlgorithm::buildOctreeParallel(queue &queue, buffer<double> &current_positions_x,
                                             buffer<double> &current_positions_y, buffer<double> &current_positions_z,
                                             buffer<double> &masses) {

    std::vector<int> nodeIsLocked_vec(16 * numberOfBodies, 0);
    buffer<int> nodeIsLocked = nodeIsLocked_vec;

    std::vector<std::size_t> nextFreeNodeID_vec(1);
    buffer<std::size_t> nextFreeNodeID = nextFreeNodeID_vec;

    std::size_t N = numberOfBodies;

    queue.submit([&](handler &h) {
        accessor<std::size_t> UPPER_NW(upper_NW, h);
        accessor<std::size_t> UPPER_NE(upper_NE, h);
        accessor<std::size_t> UPPER_SW(upper_SW, h);
        accessor<std::size_t> UPPER_SE(upper_SE, h);

        accessor<std::size_t> LOWER_NW(lower_NW, h);
        accessor<std::size_t> LOWER_NE(lower_NE, h);
        accessor<std::size_t> LOWER_SW(lower_SW, h);
        accessor<std::size_t> LOWER_SE(lower_SE, h);

        accessor<double> EDGE_LENGTHS(edgeLengths, h);

        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);

        accessor<std::size_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);

        accessor<int> NODE_LOCKED(nodeIsLocked, h);

        accessor<double> SUM_MASSES(sumOfMasses, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);

        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);

        double edgeLength = AABB_EdgeLength;
        double minX = min_x;
        double minY = min_y;
        double minZ = min_z;
        /*
        h.single_task([=]() {
            // root node 0: the AABB of all bodies
            NEXT_FREE_NODE_ID[0] = 1;

            for (int i = 0; i < 16 * N; ++i) {


                EDGE_LENGTHS[i] = edgeLength;
                MIN_X[i] = minX;
                MIN_Y[i] = minY;
                MIN_Z[i] = minZ;

                // root has no children yet.
                UPPER_NW[i] = 0;
                UPPER_NE[i] = 0;
                UPPER_SW[i] = 0;
                UPPER_SE[i] = 0;
                LOWER_NW[i] = 0;
                LOWER_NE[i] = 0;
                LOWER_SW[i] = 0;
                LOWER_SE[i] = 0;


                NODE_LOCKED[i] = 0;


                SUM_MASSES[i] = 0;
                CENTER_OF_MASS_X[i] = 0;
                CENTER_OF_MASS_Y[i] = 0;
                CENTER_OF_MASS_Z[i] = 0;

                BODY_OF_NODE[i] = N;
            }

        });
         */
        h.single_task([=]() {
            // root node 0: the AABB of all bodies
            EDGE_LENGTHS[0] = edgeLength;
            MIN_X[0] = minX;
            MIN_Y[0] = minY;
            MIN_Z[0] = minZ;

            // root has no children yet.
            UPPER_NW[0] = 0;
            UPPER_NE[0] = 0;
            UPPER_SW[0] = 0;
            UPPER_SE[0] = 0;
            LOWER_NW[0] = 0;
            LOWER_NE[0] = 0;
            LOWER_SW[0] = 0;
            LOWER_SE[0] = 0;

            NEXT_FREE_NODE_ID[0] = 1;

            NODE_LOCKED[0] = 0;


            SUM_MASSES[0] = 0;
            CENTER_OF_MASS_X[0] = 0;
            CENTER_OF_MASS_Y[0] = 0;
            CENTER_OF_MASS_Z[0] = 0;

            BODY_OF_NODE[0] = N;
        });

    }).wait();


    queue.submit([&](handler &h) {

        int tileSize = 32;// tile size should be of the form 2^x

        // global size of the nd_range kernel has to be divisible by the tile size (local size).
        // The purpose of the padding is that numberOfBodies + padding is divisible by the tile size.
        std::size_t padding = tileSize - (numberOfBodies % tileSize);

        accessor<int> NODE_LOCKED(nodeIsLocked, h);

        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);

        accessor<double> MASSES(masses, h);

        accessor<std::size_t> UPPER_NW(upper_NW, h);
        accessor<std::size_t> UPPER_NE(upper_NE, h);
        accessor<std::size_t> UPPER_SW(upper_SW, h);
        accessor<std::size_t> UPPER_SE(upper_SE, h);

        accessor<std::size_t> LOWER_NW(lower_NW, h);
        accessor<std::size_t> LOWER_NE(lower_NE, h);
        accessor<std::size_t> LOWER_SW(lower_SW, h);
        accessor<std::size_t> LOWER_SE(lower_SE, h);

        accessor<double> EDGE_LENGTHS(edgeLengths, h);

        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);

        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);

        accessor<double> SUM_MASSES(sumOfMasses, h);

        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);


        accessor<std::size_t> NEXT_FREE_NODE_ID(nextFreeNodeID, h);


        //h.parallel_for(nd_range<1>(range<1>(N), range<1>(N)), [=](auto &nd_item) {
        h.parallel_for(numberOfBodies, [=](auto &i) {

            atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                    access::address_space::global_space> nextFreeNodeIDAccessor(NEXT_FREE_NODE_ID[0]);

            //int i = nd_item.get_global_id();
            if (i < N) {

                //for (std::size_t i = 0; i < N; ++i) {




//                std::size_t currentDepth = 0;
                std::size_t currentNode = 0;


                bool nodeInserted = false;

                while (!nodeInserted) {

                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                            access::address_space::global_space> nodeIsLeafCheck(
                            UPPER_NW[currentNode]);

                    if (nodeIsLeafCheck.load(memory_order::acquire, memory_scope::device) == 0) {
                        // the current node is a leaf node

                        // We check if the current node is locked by some other thread. If it is not, this thread locks it and
                        // continues with the insertion process.
                        int exp = 0;
                        atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                                access::address_space::global_space> atomicNodeIsLockedAccessor(
                                NODE_LOCKED[currentNode]);

                        if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1, memory_order::acq_rel,
                                                                               memory_scope::device)) {

                            atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                    access::address_space::global_space> atomicBodyOfNode(
                                    BODY_OF_NODE[currentNode]);

                            if (atomicBodyOfNode.load(memory_order::acquire, memory_scope::device) == N) {
                                // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body
                                atomicBodyOfNode.store(i, memory_order::release, memory_scope::device);

                                // update sum masses and center of mass
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMasses(
                                        SUM_MASSES[currentNode]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_X(
                                        CENTER_OF_MASS_X[currentNode]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_Y(
                                        CENTER_OF_MASS_Y[currentNode]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_Z(
                                        CENTER_OF_MASS_Z[currentNode]);

                                atomicMasses.fetch_add(MASSES[i], memory_order::acq_rel, memory_scope::device);
                                atomicCenterMass_X.fetch_add(POS_X[i] * MASSES[i], memory_order::acq_rel,
                                                             memory_scope::device);
                                atomicCenterMass_Y.fetch_add(POS_Y[i] * MASSES[i], memory_order::acq_rel,
                                                             memory_scope::device);
                                atomicCenterMass_Z.fetch_add(POS_Z[i] * MASSES[i], memory_order::acq_rel,
                                                             memory_scope::device);


                                nodeInserted = true;
                            } else {
                                // the leaf node already contains a body --> split the node and insert old body

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicParentMin_x(
                                        MIN_X[currentNode]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicParentMin_y(
                                        MIN_Y[currentNode]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicParentMin_z(
                                        MIN_Z[currentNode]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicEdgeLength(
                                        EDGE_LENGTHS[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_sw(
                                        LOWER_SW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_nw(
                                        LOWER_NW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_se(
                                        LOWER_SE[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_ne(
                                        LOWER_NE[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_sw(
                                        UPPER_SW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_nw(
                                        UPPER_NW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_se(
                                        UPPER_SE[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_ne(
                                        UPPER_NE[currentNode]);


                                std::size_t bodyIDinNode = atomicBodyOfNode.load(memory_order::acquire);

                                // determine insertion index for the new child nodes and reserve 8 indices
                                std::size_t firstIndex = nextFreeNodeIDAccessor.fetch_add(8);


                                double parentEdgeLength = atomicEdgeLength.load(memory_order::acquire,
                                                                                memory_scope::device);
                                double childEdgeLength = parentEdgeLength / 2;

                                double parent_min_x = atomicParentMin_x;
                                double parent_min_y = atomicParentMin_y;
                                double parent_min_z = atomicParentMin_z;


                                // set the edge lengths of the child nodes
                                for (std::size_t idx = firstIndex; idx < firstIndex + 8; ++idx) {
                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicEdgeLengthChild(
                                            EDGE_LENGTHS[idx]);
                                    atomicEdgeLengthChild.store(childEdgeLength, memory_order::release,
                                                                memory_scope::device);
                                }

                                // create the 8 new octants
                                upper_nw.store(firstIndex, memory_order::release, memory_scope::device);
                                upper_ne.store(firstIndex + 1, memory_order::release, memory_scope::device);
                                upper_sw.store(firstIndex + 2, memory_order::release, memory_scope::device);
                                upper_se.store(firstIndex + 3, memory_order::release, memory_scope::device);
                                lower_nw.store(firstIndex + 4, memory_order::release, memory_scope::device);
                                lower_ne.store(firstIndex + 5, memory_order::release, memory_scope::device);
                                lower_sw.store(firstIndex + 6, memory_order::release, memory_scope::device);
                                lower_se.store(firstIndex + 7, memory_order::release, memory_scope::device);


                                // min x,y,z values of the upperNW child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_0(
                                        MIN_X[firstIndex]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_0(
                                        MIN_Y[firstIndex]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_0(
                                        MIN_Z[firstIndex]);

                                atomicMin_x_0.store(parent_min_x, memory_order::release, memory_scope::device);
                                atomicMin_y_0.store(parent_min_y + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_z_0.store(parent_min_z, memory_order::release, memory_scope::device);


                                // min x,y,z values of the upperNE child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_1(
                                        MIN_X[firstIndex + 1]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_1(
                                        MIN_Y[firstIndex + 1]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_1(
                                        MIN_Z[firstIndex + 1]);

                                atomicMin_x_1.store(parent_min_x + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_y_1.store(parent_min_y + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_z_1.store(parent_min_z, memory_order::release, memory_scope::device);


                                // min x,y,z values of the upperSW child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_2(
                                        MIN_X[firstIndex + 2]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_2(
                                        MIN_Y[firstIndex + 2]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_2(
                                        MIN_Z[firstIndex + 2]);

                                atomicMin_x_2.store(parent_min_x, memory_order::release, memory_scope::device);
                                atomicMin_y_2.store(parent_min_y + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_z_2.store(parent_min_z + childEdgeLength, memory_order::release,
                                                    memory_scope::device);


                                // min x,y,z values of the upperSE child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_3(
                                        MIN_X[firstIndex + 3]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_3(
                                        MIN_Y[firstIndex + 3]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_3(
                                        MIN_Z[firstIndex + 3]);

                                atomicMin_x_3.store(parent_min_x + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_y_3.store(parent_min_y + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_z_3.store(parent_min_z + childEdgeLength, memory_order::release,
                                                    memory_scope::device);


                                // min x,y,z values of the lowerNW child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_4(
                                        MIN_X[firstIndex + 4]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_4(
                                        MIN_Y[firstIndex + 4]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_4(
                                        MIN_Z[firstIndex + 4]);

                                atomicMin_x_4.store(parent_min_x, memory_order::release, memory_scope::device);
                                atomicMin_y_4.store(parent_min_y, memory_order::release, memory_scope::device);
                                atomicMin_z_4.store(parent_min_z, memory_order::release, memory_scope::device);


                                // min x,y,z values of the lowerNE child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_5(
                                        MIN_X[firstIndex + 5]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_5(
                                        MIN_Y[firstIndex + 5]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_5(
                                        MIN_Z[firstIndex + 5]);

                                atomicMin_x_5.store(parent_min_x + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_y_5.store(parent_min_y, memory_order::release, memory_scope::device);
                                atomicMin_z_5.store(parent_min_z, memory_order::release, memory_scope::device);

                                // min x,y,z values of the lowerSW child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_6(
                                        MIN_X[firstIndex + 6]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_6(
                                        MIN_Y[firstIndex + 6]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_6(
                                        MIN_Z[firstIndex + 6]);

                                atomicMin_x_6.store(parent_min_x, memory_order::release, memory_scope::device);
                                atomicMin_y_6.store(parent_min_y, memory_order::release, memory_scope::device);
                                atomicMin_z_6.store(parent_min_z + childEdgeLength, memory_order::release,
                                                    memory_scope::device);


                                // min x,y,z values of the lowerSE child node
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_x_7(
                                        MIN_X[firstIndex + 7]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_y_7(
                                        MIN_Y[firstIndex + 7]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMin_z_7(
                                        MIN_Z[firstIndex + 7]);

                                atomicMin_x_7.store(parent_min_x + childEdgeLength, memory_order::release,
                                                    memory_scope::device);
                                atomicMin_y_7.store(parent_min_y, memory_order::release, memory_scope::device);
                                atomicMin_z_7.store(parent_min_z + childEdgeLength, memory_order::release,
                                                    memory_scope::device);


                                // initially, the newly created octants will not have any children.
                                // 0 is also the root node index, but since the root will never be a child of any node, it can be used here to identify
                                // leaf nodes.
                                // Furthermore, since these nodes do not contain any bodies yet, the impossible body ID numberOfBodies gets used.
                                for (std::size_t idx = firstIndex; idx < firstIndex + 8; ++idx) {
                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicParentMin_x_i(
                                            MIN_X[idx]);
                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicParentMin_y_i(
                                            MIN_Y[idx]);
                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicParentMin_z_i(
                                            MIN_Z[idx]);
                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicEdgeLength_i(
                                            EDGE_LENGTHS[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> lower_sw_i(
                                            LOWER_SW[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> lower_nw_i(
                                            LOWER_NW[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> lower_se_i(
                                            LOWER_SE[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> lower_ne_i(
                                            LOWER_NE[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> upper_sw_i(
                                            UPPER_SW[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> upper_nw_i(
                                            UPPER_NW[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> upper_se_i(
                                            UPPER_SE[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> upper_ne_i(
                                            UPPER_NE[idx]);

                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicMasses_i(
                                            SUM_MASSES[idx]);

                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicCenterMass_X_i(
                                            CENTER_OF_MASS_X[idx]);

                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicCenterMass_Y_i(
                                            CENTER_OF_MASS_Y[idx]);

                                    atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicCenterMass_Z_i(
                                            CENTER_OF_MASS_Z[idx]);

                                    atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                            access::address_space::global_space> atomicBodyOfNode_i(
                                            BODY_OF_NODE[idx]);


                                    upper_nw_i.store(0, memory_order::release, memory_scope::device);
                                    upper_ne_i.store(0, memory_order::release, memory_scope::device);
                                    upper_sw_i.store(0, memory_order::release, memory_scope::device);
                                    upper_se_i.store(0, memory_order::release, memory_scope::device);

                                    lower_nw_i.store(0, memory_order::release, memory_scope::device);
                                    lower_ne_i.store(0, memory_order::release, memory_scope::device);
                                    lower_sw_i.store(0, memory_order::release, memory_scope::device);
                                    lower_se_i.store(0, memory_order::release, memory_scope::device);


                                    atomicBodyOfNode_i.store(N, memory_order::release, memory_scope::device);

                                    atomicCenterMass_X_i.store(0, memory_order::release, memory_scope::device);
                                    atomicCenterMass_Y_i.store(0, memory_order::release, memory_scope::device);
                                    atomicCenterMass_Z_i.store(0, memory_order::release, memory_scope::device);

                                    atomicMasses_i.store(0, memory_order::release, memory_scope::device);
                                }

                                // determine the new octant for the old body
                                std::size_t octantID;

                                bool upperPart = POS_Y[bodyIDinNode] > parent_min_y + (parentEdgeLength / 2);
                                bool rightPart = POS_X[bodyIDinNode] > parent_min_x + (parentEdgeLength / 2);
                                bool backPart = POS_Z[bodyIDinNode] < parent_min_z + (parentEdgeLength / 2);

                                if (!upperPart && !rightPart && !backPart) {
                                    octantID = lower_sw.load(memory_order::acquire, memory_scope::device);
                                } else if (!upperPart && !rightPart && backPart) {
                                    octantID = lower_nw.load(memory_order::acquire, memory_scope::device);
                                } else if (!upperPart && rightPart && !backPart) {
                                    octantID = lower_se.load(memory_order::acquire, memory_scope::device);
                                } else if (!upperPart && rightPart && backPart) {
                                    octantID = lower_ne.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && !rightPart && !backPart) {
                                    octantID = upper_sw.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && !rightPart && backPart) {
                                    octantID = upper_nw.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && rightPart && !backPart) {
                                    octantID = upper_se.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && rightPart && backPart) {
                                    octantID = upper_ne.load(memory_order::acquire, memory_scope::device);
                                }

                                // insert the old body into the new octant it belongs to and remove it from the parent node
                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicBodyOfOctant(
                                        BODY_OF_NODE[octantID]);

                                atomicBodyOfOctant.store(bodyIDinNode, memory_order::release, memory_scope::device);
                                atomicBodyOfNode.store(N, memory_order::release, memory_scope::device);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMasses(
                                        SUM_MASSES[octantID]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_X(
                                        CENTER_OF_MASS_X[octantID]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_Y(
                                        CENTER_OF_MASS_Y[octantID]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_Z(
                                        CENTER_OF_MASS_Z[octantID]);


                                atomicMasses.fetch_add(MASSES[bodyIDinNode], memory_order::acq_rel,
                                                       memory_scope::device);
                                atomicCenterMass_X.fetch_add(POS_X[bodyIDinNode] * MASSES[bodyIDinNode],
                                                             memory_order::acq_rel,
                                                             memory_scope::device);
                                atomicCenterMass_Y.fetch_add(POS_Y[bodyIDinNode] * MASSES[bodyIDinNode],
                                                             memory_order::acq_rel,
                                                             memory_scope::device);
                                atomicCenterMass_Z.fetch_add(POS_Z[bodyIDinNode] * MASSES[bodyIDinNode],
                                                             memory_order::acq_rel,
                                                             memory_scope::device);

                                // atomic_fence(memory_order::acq_rel, memory_scope::device);


                            }

                            // free the node and expose new subtree
                            atomic_fence(memory_order::acq_rel, memory_scope::device);
                            atomicNodeIsLockedAccessor.fetch_sub(1, memory_order::acq_rel, memory_scope::device);


                        }
                    } else {
                        // the current node is not a leaf node, i.e. it has 8 children
                        // --> determine the octant, the body has to be inserted and set this octant as current node.


                        int exp = 0;
                        atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                                access::address_space::global_space> atomicNodeIsLockedAccessor(
                                NODE_LOCKED[currentNode]);

                        bool descentPossible = false;
                        while (!descentPossible) {
                            if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1, memory_order::acq_rel,
                                                                                   memory_scope::device)) {
                                // if no other Thread currently works on this node

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicParentMin_x(
                                        MIN_X[currentNode]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicParentMin_y(
                                        MIN_Y[currentNode]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicParentMin_z(
                                        MIN_Z[currentNode]);
                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicEdgeLength(
                                        EDGE_LENGTHS[currentNode]);

                                std::size_t octantID;
                                //double parentMin_x = MIN_X[currentNode];
                                //double parentMin_y = MIN_Y[currentNode];
                                //double parentMin_z = MIN_Z[currentNode];
                                //double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                double parentMin_x = atomicParentMin_x.load(memory_order::acquire,
                                                                            memory_scope::device);
                                double parentMin_y = atomicParentMin_y.load(memory_order::acquire,
                                                                            memory_scope::device);
                                double parentMin_z = atomicParentMin_z.load(memory_order::acquire,
                                                                            memory_scope::device);
                                double parentEdgeLength = atomicEdgeLength.load(memory_order::acquire,
                                                                                memory_scope::device);

                                bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2);
                                bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2);
                                bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_sw(
                                        LOWER_SW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_nw(
                                        LOWER_NW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_se(
                                        LOWER_SE[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> lower_ne(
                                        LOWER_NE[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_sw(
                                        UPPER_SW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_nw(
                                        UPPER_NW[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_se(
                                        UPPER_SE[currentNode]);

                                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> upper_ne(
                                        UPPER_NE[currentNode]);


                                if (!upperPart && !rightPart && !backPart) {
                                    octantID = lower_sw.load(memory_order::acquire, memory_scope::device);
                                } else if (!upperPart && !rightPart && backPart) {
                                    octantID = lower_nw.load(memory_order::acquire, memory_scope::device);
                                } else if (!upperPart && rightPart && !backPart) {
                                    octantID = lower_se.load(memory_order::acquire, memory_scope::device);
                                } else if (!upperPart && rightPart && backPart) {
                                    octantID = lower_ne.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && !rightPart && !backPart) {
                                    octantID = upper_sw.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && !rightPart && backPart) {
                                    octantID = upper_nw.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && rightPart && !backPart) {
                                    octantID = upper_se.load(memory_order::acquire, memory_scope::device);
                                } else if (upperPart && rightPart && backPart) {
                                    octantID = upper_ne.load(memory_order::acquire, memory_scope::device);
                                }


                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicMasses(
                                        SUM_MASSES[currentNode]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_X(
                                        CENTER_OF_MASS_X[currentNode]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_Y(
                                        CENTER_OF_MASS_Y[currentNode]);

                                atomic_ref<double, memory_order::acq_rel, memory_scope::device,
                                        access::address_space::global_space> atomicCenterMass_Z(
                                        CENTER_OF_MASS_Z[currentNode]);


                                atomicMasses.fetch_add(MASSES[i], memory_order::acq_rel, memory_scope::device);
                                atomicCenterMass_X.fetch_add(POS_X[i] * MASSES[i], memory_order::acq_rel,
                                                             memory_scope::device);
                                atomicCenterMass_Y.fetch_add(POS_Y[i] * MASSES[i], memory_order::acq_rel,
                                                             memory_scope::device);
                                atomicCenterMass_Z.fetch_add(POS_Z[i] * MASSES[i], memory_order::acq_rel,
                                                             memory_scope::device);

                                currentNode = octantID;


                                descentPossible = true;
                                atomic_fence(memory_order::acq_rel, memory_scope::device);
                                atomicNodeIsLockedAccessor.fetch_sub(1, memory_order::acq_rel,
                                                                     memory_scope::device);
                            }
                        }
                    }
                }
            }
            // }
        });
    }).wait();

    host_accessor<double> testacc(sumOfMasses);
    std::cout << testacc[0] << std::endl;


}


void BarnesHutAlgorithm::computeMinMaxValuesAABB(queue &queue, buffer<double> &current_positions_x,
                                                 buffer<double> &current_positions_y,
                                                 buffer<double> &current_positions_z) {


    host_accessor<double> POS_X(current_positions_x);
    host_accessor<double> POS_Y(current_positions_y);
    host_accessor<double> POS_Z(current_positions_z);

    // update min and max values of the x,y,z coordinates
    for (std::size_t i = 0; i < numberOfBodies; ++i) {

        double current_x = POS_X[i];
        double current_y = POS_Y[i];
        double current_z = POS_Z[i];

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

void BarnesHutAlgorithm::splitNode(std::size_t nodeID, std::size_t firstIndex, host_accessor<std::size_t> UPPER_NW,
                                   host_accessor<std::size_t> UPPER_NE,
                                   host_accessor<std::size_t> UPPER_SW,
                                   host_accessor<std::size_t> UPPER_SE,
                                   host_accessor<std::size_t> LOWER_NW,
                                   host_accessor<std::size_t> LOWER_NE,
                                   host_accessor<std::size_t> LOWER_SW,
                                   host_accessor<std::size_t> LOWER_SE,
                                   host_accessor<double> EDGE_LENGTHS,
                                   host_accessor<double> MIN_X,
                                   host_accessor<double> MIN_Y,
                                   host_accessor<double> MIN_Z,
                                   host_accessor<std::size_t> BODY_OF_NODE,
                                   host_accessor<double> SUM_MASSES,
                                   host_accessor<double> CENTER_OF_MASS_X,
                                   host_accessor<double> CENTER_OF_MASS_Y,
                                   host_accessor<double> CENTER_OF_MASS_Z) {
//    host_accessor<std::size_t> UPPER_NW(upper_NW);
//    host_accessor<std::size_t> UPPER_NE(upper_NE);
//    host_accessor<std::size_t> UPPER_SW(upper_SW);
//    host_accessor<std::size_t> UPPER_SE(upper_SE);
//
//    host_accessor<std::size_t> LOWER_NW(lower_NW);
//    host_accessor<std::size_t> LOWER_NE(lower_NE);
//    host_accessor<std::size_t> LOWER_SW(lower_SW);
//    host_accessor<std::size_t> LOWER_SE(lower_SE);
//
//    host_accessor<double> EDGE_LENGTHS(edgeLengths);
//
//    host_accessor<double> MIN_X(min_x_values);
//    host_accessor<double> MIN_Y(min_y_values);
//    host_accessor<double> MIN_Z(min_z_values);
//
//    host_accessor<std::size_t> BODY_OF_NODE(bodyOfNode);
//
//    host_accessor<double> SUM_MASSES(sumOfMasses);
//
//    host_accessor<double> CENTER_OF_MASS_X(massCenters_x);
//    host_accessor<double> CENTER_OF_MASS_Y(massCenters_y);
//    host_accessor<double> CENTER_OF_MASS_Z(massCenters_z);

    double childEdgeLength = EDGE_LENGTHS[nodeID] / 2;

    double parent_min_x = MIN_X[nodeID];
    double parent_min_y = MIN_Y[nodeID];
    double parent_min_z = MIN_Z[nodeID];


    // set the edge lengths of the child nodes
    for (std::size_t i = firstIndex; i < firstIndex + 8; ++i) {
        EDGE_LENGTHS[i] = childEdgeLength;
    }

    // create the 8 new octants
    UPPER_NW[nodeID] = firstIndex;
    UPPER_NE[nodeID] = firstIndex + 1;
    UPPER_SW[nodeID] = firstIndex + 2;
    UPPER_SE[nodeID] = firstIndex + 3;

    LOWER_NW[nodeID] = firstIndex + 4;
    LOWER_NE[nodeID] = firstIndex + 5;
    LOWER_SW[nodeID] = firstIndex + 6;
    LOWER_SE[nodeID] = firstIndex + 7;


    // min x,y,z values of the upperNW child node
    MIN_X[firstIndex] = parent_min_x;
    MIN_Y[firstIndex] = parent_min_y + childEdgeLength;
    MIN_Z[firstIndex] = parent_min_z;

    // min x,y,z values of the upperNE child node
    MIN_X[firstIndex + 1] = parent_min_x + childEdgeLength;
    MIN_Y[firstIndex + 1] = parent_min_y + childEdgeLength;
    MIN_Z[firstIndex + 1] = parent_min_z;

    // min x,y,z values of the upperSW child node
    MIN_X[firstIndex + 2] = parent_min_x;
    MIN_Y[firstIndex + 2] = parent_min_y + childEdgeLength;
    MIN_Z[firstIndex + 2] = parent_min_z + childEdgeLength;

    // min x,y,z values of the upperSE child node
    MIN_X[firstIndex + 3] = parent_min_x + childEdgeLength;
    MIN_Y[firstIndex + 3] = parent_min_y + childEdgeLength;
    MIN_Z[firstIndex + 3] = parent_min_z + childEdgeLength;

    // min x,y,z values of the lowerNW child node
    MIN_X[firstIndex + 4] = parent_min_x;
    MIN_Y[firstIndex + 4] = parent_min_y;
    MIN_Z[firstIndex + 4] = parent_min_z;

    // min x,y,z values of the lowerNE child node
    MIN_X[firstIndex + 5] = parent_min_x + childEdgeLength;
    MIN_Y[firstIndex + 5] = parent_min_y;
    MIN_Z[firstIndex + 5] = parent_min_z;

    // min x,y,z values of the lowerSW child node
    MIN_X[firstIndex + 6] = parent_min_x;
    MIN_Y[firstIndex + 6] = parent_min_y;
    MIN_Z[firstIndex + 6] = parent_min_z + childEdgeLength;

    // min x,y,z values of the lowerSE child node
    MIN_X[firstIndex + 7] = parent_min_x + childEdgeLength;
    MIN_Y[firstIndex + 7] = parent_min_y;
    MIN_Z[firstIndex + 7] = parent_min_z + childEdgeLength;

    // initially, the newly created octants will not have any children.
    // 0 is also the root node index, but since the root will never be a child of any node, it can be used here to identify
    // leaf nodes.
    // Furthermore, since these nodes do not contain any bodies yet, the impossible body ID numberOfBodies gets used.
    for (std::size_t i = firstIndex; i < firstIndex + 8; ++i) {
        UPPER_NW[i] = 0;
        UPPER_NE[i] = 0;
        UPPER_SW[i] = 0;
        UPPER_SE[i] = 0;
        LOWER_NW[i] = 0;
        LOWER_NE[i] = 0;
        LOWER_SW[i] = 0;
        LOWER_SE[i] = 0;

        BODY_OF_NODE[i] = numberOfBodies;

        CENTER_OF_MASS_X[i] = 0;
        CENTER_OF_MASS_Y[i] = 0;
        CENTER_OF_MASS_Z[i] = 0;

        SUM_MASSES[i] = 0;
    }


}

std::size_t
BarnesHutAlgorithm::getOctantContainingBody(double body_position_x, double body_position_y, double body_position_z,
                                            std::size_t parentNodeID, host_accessor<std::size_t> UPPER_NW,
                                            host_accessor<std::size_t> UPPER_NE,
                                            host_accessor<std::size_t> UPPER_SW,
                                            host_accessor<std::size_t> UPPER_SE,
                                            host_accessor<std::size_t> LOWER_NW,
                                            host_accessor<std::size_t> LOWER_NE,
                                            host_accessor<std::size_t> LOWER_SW,
                                            host_accessor<std::size_t> LOWER_SE,
                                            host_accessor<double> EDGE_LENGTHS,
                                            host_accessor<double> MIN_X,
                                            host_accessor<double> MIN_Y,
                                            host_accessor<double> MIN_Z) {
//    host_accessor<std::size_t> UPPER_NW(upper_NW);
//    host_accessor<std::size_t> UPPER_NE(upper_NE);
//    host_accessor<std::size_t> UPPER_SW(upper_SW);
//    host_accessor<std::size_t> UPPER_SE(upper_SE);
//
//    host_accessor<std::size_t> LOWER_NW(lower_NW);
//    host_accessor<std::size_t> LOWER_NE(lower_NE);
//    host_accessor<std::size_t> LOWER_SW(lower_SW);
//    host_accessor<std::size_t> LOWER_SE(lower_SE);
//
//    host_accessor<double> MIN_X(min_x_values);
//    host_accessor<double> MIN_Y(min_y_values);
//    host_accessor<double> MIN_Z(min_z_values);
//
//    host_accessor<double> EDGE_LENGTHS(edgeLengths);


    double parentMin_x = MIN_X[parentNodeID];
    double parentMin_y = MIN_Y[parentNodeID];
    double parentMin_z = MIN_Z[parentNodeID];
    double parentEdgeLength = EDGE_LENGTHS[parentNodeID];

    bool upperPart = body_position_y > parentMin_y + (parentEdgeLength / 2);
    bool rightPart = body_position_x > parentMin_x + (parentEdgeLength / 2);
    bool backPart = body_position_z < parentMin_z + (parentEdgeLength / 2);

    if (!upperPart && !rightPart && !backPart) {
        return LOWER_SW[parentNodeID];
    } else if (!upperPart && !rightPart && backPart) {
        return LOWER_NW[parentNodeID];
    } else if (!upperPart && rightPart && !backPart) {
        return LOWER_SE[parentNodeID];
    } else if (!upperPart && rightPart && backPart) {
        return LOWER_NE[parentNodeID];
    } else if (upperPart && !rightPart && !backPart) {
        return UPPER_SW[parentNodeID];
    } else if (upperPart && !rightPart && backPart) {
        return UPPER_NW[parentNodeID];
    } else if (upperPart && rightPart && !backPart) {
        return UPPER_SE[parentNodeID];
    } else if (upperPart && rightPart && backPart) {
        return UPPER_NE[parentNodeID];
    } else {
        throw std::runtime_error("This state should not be reachable");
    }
}

void BarnesHutAlgorithm::computeAccelerations(queue &queue, buffer<double> &masses,
                                              buffer<double> &currentPositions_x,
                                              buffer<double> &currentPositions_y,
                                              buffer<double> &currentPositions_z,
                                              buffer<double> &acceleration_x, buffer<double> &acceleration_y,
                                              buffer<double> &acceleration_z) {

    auto begin = std::chrono::steady_clock::now();

    maxTreeDepth = 50;

    std::size_t stackSize = (8 * maxTreeDepth); // determine the stack size for each work item

    std::vector<std::size_t> nodesOnStack_vec(stackSize * numberOfBodies, bodyOfNode.size());

    buffer<std::size_t> nodesOnStack = nodesOnStack_vec;
//    std::vector<std::size_t> test(numberOfBodies);
//    buffer<std::size_t> testBuff = test;


    double epsilon_2 = pow(10, -22);


    queue.submit([&](handler &h) {

        accessor<double> POS_X(currentPositions_x, h);
        accessor<double> POS_Y(currentPositions_y, h);
        accessor<double> POS_Z(currentPositions_z, h);

        accessor<double> MASSES(masses, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);

        accessor<double> ACC_X(acceleration_x, h);
        accessor<double> ACC_Y(acceleration_y, h);
        accessor<double> ACC_Z(acceleration_z, h);

        accessor<double> EDGE_LENGTHS(edgeLengths, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);

        accessor<std::size_t> UPPER_NW(upper_NW, h);
        accessor<std::size_t> UPPER_NE(upper_NE, h);
        accessor<std::size_t> UPPER_SW(upper_SW, h);
        accessor<std::size_t> UPPER_SE(upper_SE, h);
        accessor<std::size_t> LOWER_NW(lower_NW, h);
        accessor<std::size_t> LOWER_NE(lower_NE, h);
        accessor<std::size_t> LOWER_SW(lower_SW, h);
        accessor<std::size_t> LOWER_SE(lower_SE, h);

//        accessor<std::size_t> TestACC(testBuff, h);

        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);

        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);

        accessor<std::size_t> NODES_ON_STACK(nodesOnStack, h);

        std::size_t N = numberOfBodies;
        double G = this->G;
        double THETA = 1.05;


        h.parallel_for(numberOfBodies, [=](auto &i) {
            double acc_x = 0;
            double acc_y = 0;
            double acc_z = 0;

            double pos_x_i = POS_X[i];
            double pos_y_i = POS_Y[i];
            double pos_z_i = POS_Z[i];

//            TestACC[i] = 0;


            int currentStackIndex = (stackSize * i) + 1; // start index for the stack of current work item.
            NODES_ON_STACK[currentStackIndex] = 0; // push the root node on the stack

            // currentStackIndex == (stackSize * i) --> The stack of the work item is empty
            while (currentStackIndex != (stackSize * i)) {
//                if (currentStackIndex - ((stackSize * i) + 1) > TestACC[i]) {
//                    TestACC[i] = currentStackIndex - ((stackSize * i) + 1);
//                }

                std::size_t current_Node = NODES_ON_STACK[currentStackIndex]; // get node from stack
                currentStackIndex -= 1;


                if (SUM_MASSES[current_Node] != 0 && BODY_OF_NODE[current_Node] != i) {
                    // the node is not empty and does not contain the current body.
                    double d_x = (CENTER_OF_MASS_X[current_Node] / SUM_MASSES[current_Node]) - pos_x_i;
                    double d_y = (CENTER_OF_MASS_Y[current_Node] / SUM_MASSES[current_Node]) - pos_y_i;
                    double d_z = (CENTER_OF_MASS_Z[current_Node] / SUM_MASSES[current_Node]) - pos_z_i;
                    double d = sycl::sqrt(d_x * d_x + d_y * d_y + d_z * d_z);

                    double currentTheta = EDGE_LENGTHS[current_Node] / d;
                    if (((currentTheta < THETA) || BODY_OF_NODE[current_Node] != N)) {
                        // center of mass of the node can be used to compute the acceleration

                        double denominator = sycl::sqrt((d_x * d_x) + (d_y * d_y) + (d_z * d_z) + epsilon_2);
                        denominator = denominator * denominator * denominator;

                        acc_x += SUM_MASSES[current_Node] * (d_x / denominator);
                        acc_y += SUM_MASSES[current_Node] * (d_y / denominator);
                        acc_z += SUM_MASSES[current_Node] * (d_z / denominator);

                    } else {
                        // center of mass of the node can not be used --> Push all children on the stack and continue
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = UPPER_NW[current_Node];
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = UPPER_NE[current_Node];
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = UPPER_SW[current_Node];
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = UPPER_SE[current_Node];
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = LOWER_NW[current_Node];
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = LOWER_NE[current_Node];
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = LOWER_SW[current_Node];
                        currentStackIndex += 1;
                        NODES_ON_STACK[currentStackIndex] = LOWER_SE[current_Node];
                    }
                }
            }

            ACC_X[i] = acc_x * G;
            ACC_Y[i] = acc_y * G;
            ACC_Z[i] = acc_z * G;
        });
    }).wait();

//    host_accessor TestHOST(testBuff);
//    std::size_t maximum = 0;
//    for (std::size_t n = 0; n < numberOfBodies; ++n) {
//        if (TestHOST[n] > maximum) {
//            maximum = TestHOST[n];
//        }
//
//    }
//    std::cout << "Maximum " << maximum << std::endl;

    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;

}

void BarnesHutAlgorithm::computeMasses(queue &queue, buffer<double> &current_positions_x,
                                       buffer<double> &current_positions_y, buffer<double> &current_positions_z,
                                       buffer<double> &masses) {
    maxTreeDepth = 50;

    std::size_t stackSize = (8 * maxTreeDepth); // determine the stack size for each work item

    std::vector<std::size_t> nodesOnStack_vec(stackSize * numberOfBodies, bodyOfNode.size());

    buffer<std::size_t> nodesOnStack = nodesOnStack_vec;


    queue.submit([&](handler &h) {

        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);

        accessor<double> MASSES(masses, h);
        accessor<double> SUM_MASSES(sumOfMasses, h);

        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);

        accessor<std::size_t> UPPER_NW(upper_NW, h);
        accessor<std::size_t> UPPER_NE(upper_NE, h);
        accessor<std::size_t> UPPER_SW(upper_SW, h);
        accessor<std::size_t> UPPER_SE(upper_SE, h);
        accessor<std::size_t> LOWER_NW(lower_NW, h);
        accessor<std::size_t> LOWER_NE(lower_NE, h);
        accessor<std::size_t> LOWER_SW(lower_SW, h);
        accessor<std::size_t> LOWER_SE(lower_SE, h);


        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);

        accessor<std::size_t> NODES_ON_STACK(nodesOnStack, h);

        std::size_t N = numberOfBodies;


        h.single_task([=]() {

        });
    }).wait();

}
