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
          centerOfMass_z_vec(16 * numberOfBodies),
          subtreeOfBodyID_vec(numberOfBodies),
          nodeIsLocked_vec(16 * numberOfBodies, 0){
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
    buildOctreeParallel2(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z, masses);

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

        auto begin = std::chrono::steady_clock::now();
        resetOctree();
        computeMinMaxValuesAABB(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z);
        buildOctreeParallel2(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z,
                             masses);

        computeAccelerations(queue, masses, intermediatePosition_x, intermediatePosition_y,
                             intermediatePosition_z,
                             acceleration_x, acceleration_y, acceleration_z);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Time of step:  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
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


    int splitLevel = 3; // there will 8^splitLevel subtrees
    int subTreeCount = pow(8, splitLevel);
    int sliceSize = pow(4, splitLevel);
    int rowSize = pow(2, splitLevel);

    std::size_t N = numberOfBodies;

    std::size_t firstIndex = (8 * subTreeCount - 1) / 7; // first index where the subtree information can be stored

    double edgeLengthAtSplitLevel = AABB_EdgeLength / pow(2, splitLevel);

    std::vector<std::size_t> subtreeRootIndex_vec(subTreeCount, 0);
    buffer<std::size_t> subtreeRootIndex = subtreeRootIndex_vec;

    std::vector<std::size_t> maxTreeDepth_vec(1, splitLevel);
    buffer<std::size_t> maxTreeDepths = maxTreeDepth_vec;

    std::vector<std::size_t> bodyCountSubtree_vec(firstIndex, 0);
    buffer<std::size_t> bodyCountSubtree = bodyCountSubtree_vec;



    // build tree to subTree level
    queue.submit([&](handler &h) {
        accessor<int> SUBTREE_OF_BODY(subtreeOfBodyID, h);

        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);

        accessor<std::size_t> ROOT_SUBTREE(subtreeRootIndex, h);

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

        double edgeLength = AABB_EdgeLength;
        double minX = min_x;
        double minY = min_y;
        double minZ = min_z;

        accessor<double> SUM_MASSES(sumOfMasses, h);

        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);


        h.single_task([=]() {
            std::size_t firstIndexOnLevel = 0;
            int nextFreeIndex = 1;

            // values for root node
            EDGE_LENGTHS[0] = edgeLength;
            MIN_X[0] = minX;
            MIN_Y[0] = minY;
            MIN_Z[0] = minZ;

            for (int i = 0; i < splitLevel; ++i) {
                // iterate over each level


                for (std::size_t j = firstIndexOnLevel; j < firstIndexOnLevel + sycl::pow(8.0, (double) i); ++j) {
                    // iterate over each node on current level and split the node

                    double childEdgeLength = EDGE_LENGTHS[j] / 2;

                    double parent_min_x = MIN_X[j];
                    double parent_min_y = MIN_Y[j];
                    double parent_min_z = MIN_Z[j];


                    // set the edge lengths of the child nodes
                    for (std::size_t idx = nextFreeIndex; idx < nextFreeIndex + 8; ++idx) {
                        EDGE_LENGTHS[idx] = childEdgeLength;
                    }

                    // create the 8 new octants
                    UPPER_NW[j] = nextFreeIndex;
                    UPPER_NE[j] = nextFreeIndex + 1;
                    UPPER_SW[j] = nextFreeIndex + 2;
                    UPPER_SE[j] = nextFreeIndex + 3;

                    LOWER_NW[j] = nextFreeIndex + 4;
                    LOWER_NE[j] = nextFreeIndex + 5;
                    LOWER_SW[j] = nextFreeIndex + 6;
                    LOWER_SE[j] = nextFreeIndex + 7;


                    // min x,y,z values of the upperNW child node
                    MIN_X[nextFreeIndex] = parent_min_x;
                    MIN_Y[nextFreeIndex] = parent_min_y + childEdgeLength;
                    MIN_Z[nextFreeIndex] = parent_min_z;

                    // min x,y,z values of the upperNE child node
                    MIN_X[nextFreeIndex + 1] = parent_min_x + childEdgeLength;
                    MIN_Y[nextFreeIndex + 1] = parent_min_y + childEdgeLength;
                    MIN_Z[nextFreeIndex + 1] = parent_min_z;

                    // min x,y,z values of the upperSW child node
                    MIN_X[nextFreeIndex + 2] = parent_min_x;
                    MIN_Y[nextFreeIndex + 2] = parent_min_y + childEdgeLength;
                    MIN_Z[nextFreeIndex + 2] = parent_min_z + childEdgeLength;

                    // min x,y,z values of the upperSE child node
                    MIN_X[nextFreeIndex + 3] = parent_min_x + childEdgeLength;
                    MIN_Y[nextFreeIndex + 3] = parent_min_y + childEdgeLength;
                    MIN_Z[nextFreeIndex + 3] = parent_min_z + childEdgeLength;

                    // min x,y,z values of the lowerNW child node
                    MIN_X[nextFreeIndex + 4] = parent_min_x;
                    MIN_Y[nextFreeIndex + 4] = parent_min_y;
                    MIN_Z[nextFreeIndex + 4] = parent_min_z;

                    // min x,y,z values of the lowerNE child node
                    MIN_X[nextFreeIndex + 5] = parent_min_x + childEdgeLength;
                    MIN_Y[nextFreeIndex + 5] = parent_min_y;
                    MIN_Z[nextFreeIndex + 5] = parent_min_z;

                    // min x,y,z values of the lowerSW child node
                    MIN_X[nextFreeIndex + 6] = parent_min_x;
                    MIN_Y[nextFreeIndex + 6] = parent_min_y;
                    MIN_Z[nextFreeIndex + 6] = parent_min_z + childEdgeLength;

                    // min x,y,z values of the lowerSE child node
                    MIN_X[nextFreeIndex + 7] = parent_min_x + childEdgeLength;
                    MIN_Y[nextFreeIndex + 7] = parent_min_y;
                    MIN_Z[nextFreeIndex + 7] = parent_min_z + childEdgeLength;

                    // initially, the newly created octants will not have any children.
                    // 0 is also the root node index, but since the root will never be a child of any node, it can be used here to identify
                    // leaf nodes.
                    // Furthermore, since these nodes do not contain any bodies yet, the impossible body ID numberOfBodies gets used.
                    for (std::size_t idx = nextFreeIndex; idx < nextFreeIndex + 8; ++idx) {
                        UPPER_NW[idx] = 0;
                        UPPER_NE[idx] = 0;
                        UPPER_SW[idx] = 0;
                        UPPER_SE[idx] = 0;
                        LOWER_NW[idx] = 0;
                        LOWER_NE[idx] = 0;
                        LOWER_SW[idx] = 0;
                        LOWER_SE[idx] = 0;

                        BODY_OF_NODE[idx] = N;

                        CENTER_OF_MASS_X[idx] = 0;
                        CENTER_OF_MASS_Y[idx] = 0;
                        CENTER_OF_MASS_Z[idx] = 0;

                        SUM_MASSES[idx] = 0;
                    }

                    nextFreeIndex += 8;

                }


                if (i < splitLevel) {
                    firstIndexOnLevel += sycl::pow(8.0, (double) i);
                }

            }
            int idx = 0;
            for (std::size_t i = firstIndexOnLevel; i < firstIndexOnLevel + sycl::pow(8.0, (double) splitLevel); ++i) {
                // iterate over all nodes on final level
                ROOT_SUBTREE[idx] = i;
                idx++;
            }


        });
    }).wait();




    // determine the octree for each body
    queue.submit([&](handler &h) {
        accessor<int> SUBTREE_OF_BODY(subtreeOfBodyID, h);

        accessor<double> POS_X(current_positions_x, h);
        accessor<double> POS_Y(current_positions_y, h);
        accessor<double> POS_Z(current_positions_z, h);


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

        accessor<std::size_t> BODY_COUNT(bodyCountSubtree, h);

        // determine the subtree for each body
        h.parallel_for(numberOfBodies, [=](auto &i) {


            std::size_t parentNodeID = 0;

            for (int level = 0; level < splitLevel; ++level) {
                double parentMin_x = MIN_X[parentNodeID];
                double parentMin_y = MIN_Y[parentNodeID];
                double parentMin_z = MIN_Z[parentNodeID];
                double parentEdgeLength = EDGE_LENGTHS[parentNodeID];

                bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2);
                bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2);
                bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2);

                if (!upperPart && !rightPart && !backPart) {
                    parentNodeID = LOWER_SW[parentNodeID];
                } else if (!upperPart && !rightPart && backPart) {
                    parentNodeID = LOWER_NW[parentNodeID];
                } else if (!upperPart && rightPart && !backPart) {
                    parentNodeID = LOWER_SE[parentNodeID];
                } else if (!upperPart && rightPart && backPart) {
                    parentNodeID = LOWER_NE[parentNodeID];
                } else if (upperPart && !rightPart && !backPart) {
                    parentNodeID = UPPER_SW[parentNodeID];
                } else if (upperPart && !rightPart && backPart) {
                    parentNodeID = UPPER_NW[parentNodeID];
                } else if (upperPart && rightPart && !backPart) {
                    parentNodeID = UPPER_SE[parentNodeID];
                } else if (upperPart && rightPart && backPart) {
                    parentNodeID = UPPER_NE[parentNodeID];
                }

                atomic_ref<std::size_t, memory_order::relaxed, memory_scope::device,
                        access::address_space::global_space> incrementBodyRef(BODY_COUNT[parentNodeID]);
                incrementBodyRef.fetch_add(1);
            }

            std::size_t subtreeIndex = parentNodeID;
            SUBTREE_OF_BODY[i] = subtreeIndex;

        });
    }).wait();


    std::vector<std::size_t> nextFreeNodeID_vec(1, firstIndex);
    buffer<std::size_t> nextFreeNodeID_buf = nextFreeNodeID_vec;

//    {
//        host_accessor TEST(bodyCountSubtree);
//        for (int i = 0; i < subTreeCount; ++i) {
//            if (TEST[i + firstIndex - pow(8, splitLevel)]) {
//                std::cout << TEST[i + firstIndex - pow(8, splitLevel)] << std::endl;
//            }
//        }
//    }


    auto begin = std::chrono::steady_clock::now();


    queue.submit([&](handler &h) {
        accessor<std::size_t> NEXT_FREE_NODE_ID(nextFreeNodeID_buf, h);
        accessor<std::size_t> MAX_TREE_DEPTH(maxTreeDepths, h);

        accessor<int> SUBTREE_OF_BODY(subtreeOfBodyID, h);
        accessor<std::size_t> ROOT_SUBTREE(subtreeRootIndex, h);

        accessor<std::size_t> BODY_COUNT(bodyCountSubtree, h);

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


        maxTreeDepth = splitLevel;
        std::size_t maxTreeDepthTemp = maxTreeDepth;

        std::size_t subTreeLevelIndex = firstIndex - pow(8, splitLevel);

        h.parallel_for(subTreeCount, [=](auto &subTreeID) {
//        h.parallel_for(nd_range<1>(range<1>(subTreeCount), range<1>(32)), [=](auto &nd_item) {


//            std::size_t subTreeID = nd_item.get_global_id();
            if (BODY_COUNT[subTreeID + subTreeLevelIndex] != 0) {

                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                        access::address_space::global_space> nextFreeNodeIDAccessor(NEXT_FREE_NODE_ID[0]);

                atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                        access::address_space::global_space> max_tree_depth(MAX_TREE_DEPTH[0]);


                for (std::size_t i = 0; i < N; ++i) {
                    if (SUBTREE_OF_BODY[i] == subTreeID + subTreeLevelIndex) {
                        // if the current body belongs to this subtree insert the body

                        std::size_t currentDepth = splitLevel;
                        std::size_t currentNode = subTreeID + subTreeLevelIndex;
                        bool nodeInserted = false;
                        while (!nodeInserted) {
                            if (UPPER_NW[currentNode] == 0) {
                                // the current node is a leaf node
                                if (BODY_OF_NODE[currentNode] == N) {
                                    // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body
                                    BODY_OF_NODE[currentNode] = i;

                                    // update sum masses and center of mass
                                    SUM_MASSES[currentNode] += MASSES[i];
                                    CENTER_OF_MASS_X[currentNode] += POS_X[i] * MASSES[i];
                                    CENTER_OF_MASS_Y[currentNode] += POS_Y[i] * MASSES[i];
                                    CENTER_OF_MASS_Z[currentNode] += POS_Z[i] * MASSES[i];

//                                if (currentDepth > m) {
//                                    maxTreeDepth = currentDepth;
//                                }
                                    max_tree_depth.fetch_max(currentDepth);
                                    nodeInserted = true;
                                } else {
                                    // the leaf node already contains a body --> split the node and insert old body
                                    std::size_t nextFreeNodeID = nextFreeNodeIDAccessor.fetch_add(8);


                                    std::size_t bodyIDinNode = BODY_OF_NODE[currentNode];

                                    // split the node
                                    double childEdgeLength = EDGE_LENGTHS[currentNode] / 2;

                                    double parent_min_x = MIN_X[currentNode];
                                    double parent_min_y = MIN_Y[currentNode];
                                    double parent_min_z = MIN_Z[currentNode];


                                    // set the edge lengths of the child nodes
                                    for (std::size_t idx = nextFreeNodeID; idx < nextFreeNodeID + 8; ++idx) {
                                        EDGE_LENGTHS[idx] = childEdgeLength;
                                    }

                                    // create the 8 new octants
                                    UPPER_NW[currentNode] = nextFreeNodeID;
                                    UPPER_NE[currentNode] = nextFreeNodeID + 1;
                                    UPPER_SW[currentNode] = nextFreeNodeID + 2;
                                    UPPER_SE[currentNode] = nextFreeNodeID + 3;

                                    LOWER_NW[currentNode] = nextFreeNodeID + 4;
                                    LOWER_NE[currentNode] = nextFreeNodeID + 5;
                                    LOWER_SW[currentNode] = nextFreeNodeID + 6;
                                    LOWER_SE[currentNode] = nextFreeNodeID + 7;


                                    // min x,y,z values of the upperNW child node
                                    MIN_X[nextFreeNodeID] = parent_min_x;
                                    MIN_Y[nextFreeNodeID] = parent_min_y + childEdgeLength;
                                    MIN_Z[nextFreeNodeID] = parent_min_z;

                                    // min x,y,z values of the upperNE child node
                                    MIN_X[nextFreeNodeID + 1] = parent_min_x + childEdgeLength;
                                    MIN_Y[nextFreeNodeID + 1] = parent_min_y + childEdgeLength;
                                    MIN_Z[nextFreeNodeID + 1] = parent_min_z;

                                    // min x,y,z values of the upperSW child node
                                    MIN_X[nextFreeNodeID + 2] = parent_min_x;
                                    MIN_Y[nextFreeNodeID + 2] = parent_min_y + childEdgeLength;
                                    MIN_Z[nextFreeNodeID + 2] = parent_min_z + childEdgeLength;

                                    // min x,y,z values of the upperSE child node
                                    MIN_X[nextFreeNodeID + 3] = parent_min_x + childEdgeLength;
                                    MIN_Y[nextFreeNodeID + 3] = parent_min_y + childEdgeLength;
                                    MIN_Z[nextFreeNodeID + 3] = parent_min_z + childEdgeLength;

                                    // min x,y,z values of the lowerNW child node
                                    MIN_X[nextFreeNodeID + 4] = parent_min_x;
                                    MIN_Y[nextFreeNodeID + 4] = parent_min_y;
                                    MIN_Z[nextFreeNodeID + 4] = parent_min_z;

                                    // min x,y,z values of the lowerNE child node
                                    MIN_X[nextFreeNodeID + 5] = parent_min_x + childEdgeLength;
                                    MIN_Y[nextFreeNodeID + 5] = parent_min_y;
                                    MIN_Z[nextFreeNodeID + 5] = parent_min_z;

                                    // min x,y,z values of the lowerSW child node
                                    MIN_X[nextFreeNodeID + 6] = parent_min_x;
                                    MIN_Y[nextFreeNodeID + 6] = parent_min_y;
                                    MIN_Z[nextFreeNodeID + 6] = parent_min_z + childEdgeLength;

                                    // min x,y,z values of the lowerSE child node
                                    MIN_X[nextFreeNodeID + 7] = parent_min_x + childEdgeLength;
                                    MIN_Y[nextFreeNodeID + 7] = parent_min_y;
                                    MIN_Z[nextFreeNodeID + 7] = parent_min_z + childEdgeLength;

                                    // initially, the newly created octants will not have any children.
                                    // 0 is also the root node index, but since the root will never be a child of any node, it can be used here to identify
                                    // leaf nodes.
                                    // Furthermore, since these nodes do not contain any bodies yet, the impossible body ID numberOfBodies gets used.
                                    for (std::size_t idx = nextFreeNodeID; idx < nextFreeNodeID + 8; ++idx) {
                                        UPPER_NW[idx] = 0;
                                        UPPER_NE[idx] = 0;
                                        UPPER_SW[idx] = 0;
                                        UPPER_SE[idx] = 0;
                                        LOWER_NW[idx] = 0;
                                        LOWER_NE[idx] = 0;
                                        LOWER_SW[idx] = 0;
                                        LOWER_SE[idx] = 0;

                                        BODY_OF_NODE[idx] = N;

                                        CENTER_OF_MASS_X[idx] = 0;
                                        CENTER_OF_MASS_Y[idx] = 0;
                                        CENTER_OF_MASS_Z[idx] = 0;

                                        SUM_MASSES[idx] = 0;
                                    }

//                                nextFreeNodeID += 8;


                                    // determine the new octant for the old body
                                    std::size_t octantID;
                                    double parentMin_x = MIN_X[currentNode];
                                    double parentMin_y = MIN_Y[currentNode];
                                    double parentMin_z = MIN_Z[currentNode];
                                    double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                    bool upperPart = POS_Y[bodyIDinNode] > parentMin_y + (parentEdgeLength / 2);
                                    bool rightPart = POS_X[bodyIDinNode] > parentMin_x + (parentEdgeLength / 2);
                                    bool backPart = POS_Z[bodyIDinNode] < parentMin_z + (parentEdgeLength / 2);

                                    if (!upperPart && !rightPart && !backPart) {
                                        octantID = LOWER_SW[currentNode];
                                    } else if (!upperPart && !rightPart && backPart) {
                                        octantID = LOWER_NW[currentNode];
                                    } else if (!upperPart && rightPart && !backPart) {
                                        octantID = LOWER_SE[currentNode];
                                    } else if (!upperPart && rightPart && backPart) {
                                        octantID = LOWER_NE[currentNode];
                                    } else if (upperPart && !rightPart && !backPart) {
                                        octantID = UPPER_SW[currentNode];
                                    } else if (upperPart && !rightPart && backPart) {
                                        octantID = UPPER_NW[currentNode];
                                    } else if (upperPart && rightPart && !backPart) {
                                        octantID = UPPER_SE[currentNode];
                                    } else if (upperPart && rightPart && backPart) {
                                        octantID = UPPER_NE[currentNode];
                                    }

                                    // insert the old body into the new octant it belongs to and remove it from the parent node
                                    BODY_OF_NODE[octantID] = bodyIDinNode;
                                    BODY_OF_NODE[currentNode] = N;

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

                                std::size_t octantID;
                                double parentMin_x = MIN_X[currentNode];
                                double parentMin_y = MIN_Y[currentNode];
                                double parentMin_z = MIN_Z[currentNode];
                                double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2);
                                bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2);
                                bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2);

                                if (!upperPart && !rightPart && !backPart) {
                                    octantID = LOWER_SW[currentNode];
                                } else if (!upperPart && !rightPart && backPart) {
                                    octantID = LOWER_NW[currentNode];
                                } else if (!upperPart && rightPart && !backPart) {
                                    octantID = LOWER_SE[currentNode];
                                } else if (!upperPart && rightPart && backPart) {
                                    octantID = LOWER_NE[currentNode];
                                } else if (upperPart && !rightPart && !backPart) {
                                    octantID = UPPER_SW[currentNode];
                                } else if (upperPart && !rightPart && backPart) {
                                    octantID = UPPER_NW[currentNode];
                                } else if (upperPart && rightPart && !backPart) {
                                    octantID = UPPER_SE[currentNode];
                                } else if (upperPart && rightPart && backPart) {
                                    octantID = UPPER_NE[currentNode];
                                }

                                currentNode = octantID;
                                currentDepth += 1;

                            }

                        }
                    }
                }

            }

        });
    }).wait();

//    {
//        host_accessor Test(nextFreeNodeID_buf);
//        std::cout << "max ---------- " << Test[0] <<std::endl;
//    }

    host_accessor maxTreeDepthAccessor(maxTreeDepths);
    maxTreeDepth = maxTreeDepthAccessor[0];


    auto end = std::chrono::steady_clock::now();
    std::cout << "---------------------------------------------------------- "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;


    queue.submit([&](handler &h) {
        accessor<int> SUBTREE_OF_BODY(subtreeOfBodyID, h);
        accessor<std::size_t> BODY_OF_NODE(bodyOfNode, h);


        accessor<std::size_t> UPPER_NW(upper_NW, h);
        accessor<std::size_t> UPPER_NE(upper_NE, h);
        accessor<std::size_t> UPPER_SW(upper_SW, h);
        accessor<std::size_t> UPPER_SE(upper_SE, h);

        accessor<std::size_t> LOWER_NW(lower_NW, h);
        accessor<std::size_t> LOWER_NE(lower_NE, h);
        accessor<std::size_t> LOWER_SW(lower_SW, h);
        accessor<std::size_t> LOWER_SE(lower_SE, h);

        accessor<std::size_t> BODY_COUNT(bodyCountSubtree, h);

        accessor<double> EDGE_LENGTHS(edgeLengths, h);

        accessor<double> MIN_X(min_x_values, h);
        accessor<double> MIN_Y(min_y_values, h);
        accessor<double> MIN_Z(min_z_values, h);

        accessor<double> SUM_MASSES(sumOfMasses, h);

        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);

//       host_accessor<int> SUBTREE_OF_BODY(subtreeOfBodyID);
//       host_accessor<std::size_t> BODY_OF_NODE(bodyOfNode );
//
//
//       host_accessor<std::size_t> UPPER_NW(upper_NW );
//       host_accessor<std::size_t> UPPER_NE(upper_NE );
//       host_accessor<std::size_t> UPPER_SW(upper_SW );
//       host_accessor<std::size_t> UPPER_SE(upper_SE );
//
//       host_accessor<std::size_t> LOWER_NW(lower_NW );
//       host_accessor<std::size_t> LOWER_NE(lower_NE );
//       host_accessor<std::size_t> LOWER_SW(lower_SW );
//       host_accessor<std::size_t> LOWER_SE(lower_SE );
//
//       host_accessor<std::size_t> BODY_COUNT(bodyCountSubtree );
//
//       host_accessor<double> EDGE_LENGTHS(edgeLengths);
//
//       host_accessor<double> MIN_X(min_x_values );
//       host_accessor<double> MIN_Y(min_y_values );
//       host_accessor<double> MIN_Z(min_z_values );
//
//       host_accessor<double> SUM_MASSES(sumOfMasses );
//
//       host_accessor<double> CENTER_OF_MASS_X(massCenters_x );
//       host_accessor<double> CENTER_OF_MASS_Y(massCenters_y );
//       host_accessor<double> CENTER_OF_MASS_Z(massCenters_z );

        // determine the subtree for each body
        h.single_task([=]() {
            std::size_t firstIndexOnLevel = firstIndex;

            for (int i = splitLevel; i > 0; --i) {
                // iterate over each level of top part of tree
                firstIndexOnLevel = firstIndexOnLevel - (sycl::pow(8.0, (double) i));


                int parentCounter = 0;
                for (std::size_t j = firstIndexOnLevel; j < firstIndexOnLevel + sycl::pow(8.0, (double) i); j += 8) {
                    // iterate over each node on current level
                    std::size_t subTreesWithNodes = 0;
                    double sumMassesChildren = 0;
                    double centerMassChildren_x = 0;
                    double centerMassChildren_y = 0;
                    double centerMassChildren_z = 0;

                    for (std::size_t idx = j; idx < j + 8; ++idx) {
                        // iterate over children of one Node:
                        if (BODY_COUNT[idx] != 0) {
                            subTreesWithNodes += 1;
                        }

                        sumMassesChildren += SUM_MASSES[idx];
                        centerMassChildren_x += CENTER_OF_MASS_X[idx];
                        centerMassChildren_y += CENTER_OF_MASS_Y[idx];
                        centerMassChildren_z += CENTER_OF_MASS_Z[idx];
                    }

                    std::size_t parentID = firstIndexOnLevel - (sycl::pow(8.0, (double) i - 1)) + parentCounter;

                    SUM_MASSES[parentID] = sumMassesChildren;
                    CENTER_OF_MASS_X[parentID] = centerMassChildren_x;
                    CENTER_OF_MASS_Y[parentID] = centerMassChildren_y;
                    CENTER_OF_MASS_Z[parentID] = centerMassChildren_z;

                    if (subTreesWithNodes == 0) {
                        UPPER_NW[parentID] = 0;
                        UPPER_NE[parentID] = 0;
                        UPPER_SW[parentID] = 0;
                        UPPER_SE[parentID] = 0;
                        LOWER_NW[parentID] = 0;
                        LOWER_NE[parentID] = 0;
                        LOWER_SW[parentID] = 0;
                        LOWER_SE[parentID] = 0;
                    } else if (subTreesWithNodes == 1) {
                        std::size_t subTreeWithBody;
                        for (std::size_t idx = j; idx < j + 8; ++idx) {
                            if (BODY_COUNT[idx] != 0) {
                                subTreeWithBody = idx;
                            }
                        }
                        if (BODY_COUNT[subTreeWithBody] == 1) {
                            UPPER_NW[parentID] = 0;
                            UPPER_NE[parentID] = 0;
                            UPPER_SW[parentID] = 0;
                            UPPER_SE[parentID] = 0;
                            LOWER_NW[parentID] = 0;
                            LOWER_NE[parentID] = 0;
                            LOWER_SW[parentID] = 0;
                            LOWER_SE[parentID] = 0;

                            BODY_OF_NODE[parentID] = BODY_OF_NODE[subTreeWithBody];
                            BODY_OF_NODE[subTreeWithBody] = N;
                        }
                    }

                    parentCounter++;
                }

            }


        });
    }).wait();


}

void BarnesHutAlgorithm::buildOctreeParallel2(queue &queue, buffer<double> &current_positions_x,
                                              buffer<double> &current_positions_y, buffer<double> &current_positions_z,
                                              buffer<double> &masses) {
    auto begin = std::chrono::steady_clock::now();

    std::vector<std::size_t> maxTreeDepth_vec(1, 0);
    buffer<std::size_t> maxTreeDepths = maxTreeDepth_vec;

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

//        int tileSize = 32;// tile size should be of the form 2^x
//
//        // global size of the nd_range kernel has to be divisible by the tile size (local size).
//        // The purpose of the padding is that numberOfBodies + padding is divisible by the tile size.
//        std::size_t padding = tileSize - (numberOfBodies % tileSize);

        accessor<int> NODE_LOCKED(nodeIsLocked, h);
        accessor<std::size_t> MAX_TREE_DEPTH(maxTreeDepths, h);

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
        std::size_t bodyCountPerWorkItem;
        int workItemCount = 256;
        if (numberOfBodies > workItemCount) {
            bodyCountPerWorkItem = ceil((double) numberOfBodies / (double) workItemCount);
        } else {
            bodyCountPerWorkItem = 1;
        }

        h.parallel_for(nd_range<1>(range<1>(workItemCount), range<1>(workItemCount)), [=](auto &nd_item) {
//        h.parallel_for(numberOfBodies, [=](auto &i) {

            atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                    access::address_space::global_space> nextFreeNodeIDAccessor(NEXT_FREE_NODE_ID[0]);

            atomic_ref<std::size_t, memory_order::acq_rel, memory_scope::device,
                    access::address_space::global_space> max_tree_depth(MAX_TREE_DEPTH[0]);

//            int i = nd_item.get_global_id();
            for (std::size_t i = nd_item.get_global_id() * bodyCountPerWorkItem;
                 i < nd_item.get_global_id() * bodyCountPerWorkItem + bodyCountPerWorkItem; ++i) {
                if (i < N) {
                    std::size_t currentDepth = 0;
                    std::size_t currentNode = 0;


                    bool nodeInserted = false;

                    while (!nodeInserted) {
                        // We check if the current node is locked by some other thread. If it is not, this thread locks it and
                        // continues with the insertion process.
                        int exp = 0;
                        atomic_ref<int, memory_order::acq_rel, memory_scope::device,
                                access::address_space::global_space> atomicNodeIsLockedAccessor(
                                NODE_LOCKED[currentNode]);

                        if (atomicNodeIsLockedAccessor.compare_exchange_strong(exp, 1, memory_order::acq_rel,
                                                                               memory_scope::device)) {

                            if (UPPER_NW[currentNode] == 0) {
                                // the current node is a leaf node

                                if (BODY_OF_NODE[currentNode] == N) {
                                    // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body

                                    BODY_OF_NODE[currentNode] = i;

                                    // update sum masses and center of mass
                                    SUM_MASSES[currentNode] += MASSES[i];
                                    CENTER_OF_MASS_X[currentNode] += POS_X[i] * MASSES[i];
                                    CENTER_OF_MASS_Y[currentNode] += POS_Y[i] * MASSES[i];
                                    CENTER_OF_MASS_Z[currentNode] += POS_Z[i] * MASSES[i];

                                    max_tree_depth.fetch_max(currentDepth);
                                    nodeInserted = true;
                                } else {
                                    // the leaf node already contains a body --> split the node and insert old body

                                    std::size_t bodyIDinNode = BODY_OF_NODE[currentNode];

                                    // determine insertion index for the new child nodes and reserve 8 indices
                                    std::size_t firstIndex = nextFreeNodeIDAccessor.fetch_add(8);


                                    double childEdgeLength = EDGE_LENGTHS[currentNode] / 2;

                                    double parent_min_x = MIN_X[currentNode];
                                    double parent_min_y = MIN_Y[currentNode];
                                    double parent_min_z = MIN_Z[currentNode];


                                    // set the edge lengths of the child nodes
                                    for (std::size_t idx = firstIndex; idx < firstIndex + 8; ++idx) {
                                        EDGE_LENGTHS[idx] = childEdgeLength;
                                    }

                                    // create the 8 new octants
                                    UPPER_NW[currentNode] = firstIndex;
                                    UPPER_NE[currentNode] = firstIndex + 1;
                                    UPPER_SW[currentNode] = firstIndex + 2;
                                    UPPER_SE[currentNode] = firstIndex + 3;

                                    LOWER_NW[currentNode] = firstIndex + 4;
                                    LOWER_NE[currentNode] = firstIndex + 5;
                                    LOWER_SW[currentNode] = firstIndex + 6;
                                    LOWER_SE[currentNode] = firstIndex + 7;


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
                                    for (std::size_t idx = firstIndex; idx < firstIndex + 8; ++idx) {
                                        UPPER_NW[idx] = 0;
                                        UPPER_NE[idx] = 0;
                                        UPPER_SW[idx] = 0;
                                        UPPER_SE[idx] = 0;
                                        LOWER_NW[idx] = 0;
                                        LOWER_NE[idx] = 0;
                                        LOWER_SW[idx] = 0;
                                        LOWER_SE[idx] = 0;

                                        BODY_OF_NODE[idx] = N;

                                        CENTER_OF_MASS_X[idx] = 0;
                                        CENTER_OF_MASS_Y[idx] = 0;
                                        CENTER_OF_MASS_Z[idx] = 0;

                                        SUM_MASSES[idx] = 0;
                                    }

                                    // determine the new octant for the old body
                                    std::size_t octantID;
                                    double parentMin_x = MIN_X[currentNode];
                                    double parentMin_y = MIN_Y[currentNode];
                                    double parentMin_z = MIN_Z[currentNode];
                                    double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                    bool upperPart = POS_Y[bodyIDinNode] > parentMin_y + (parentEdgeLength / 2);
                                    bool rightPart = POS_X[bodyIDinNode] > parentMin_x + (parentEdgeLength / 2);
                                    bool backPart = POS_Z[bodyIDinNode] < parentMin_z + (parentEdgeLength / 2);

                                    if (!upperPart && !rightPart && !backPart) {
                                        octantID = LOWER_SW[currentNode];
                                    } else if (!upperPart && !rightPart && backPart) {
                                        octantID = LOWER_NW[currentNode];
                                    } else if (!upperPart && rightPart && !backPart) {
                                        octantID = LOWER_SE[currentNode];
                                    } else if (!upperPart && rightPart && backPart) {
                                        octantID = LOWER_NE[currentNode];
                                    } else if (upperPart && !rightPart && !backPart) {
                                        octantID = UPPER_SW[currentNode];
                                    } else if (upperPart && !rightPart && backPart) {
                                        octantID = UPPER_NW[currentNode];
                                    } else if (upperPart && rightPart && !backPart) {
                                        octantID = UPPER_SE[currentNode];
                                    } else if (upperPart && rightPart && backPart) {
                                        octantID = UPPER_NE[currentNode];
                                    }

                                    // insert the old body into the new octant it belongs to and remove it from the parent node
                                    BODY_OF_NODE[octantID] = bodyIDinNode;
                                    BODY_OF_NODE[currentNode] = N;

                                    SUM_MASSES[octantID] += MASSES[bodyIDinNode];
                                    CENTER_OF_MASS_X[octantID] += POS_X[bodyIDinNode] * MASSES[bodyIDinNode];
                                    CENTER_OF_MASS_Y[octantID] += POS_Y[bodyIDinNode] * MASSES[bodyIDinNode];
                                    CENTER_OF_MASS_Z[octantID] += POS_Z[bodyIDinNode] * MASSES[bodyIDinNode];

                                }

                            } else {
                                // the current node is not a leaf node, i.e. it has 8 children
                                // --> determine the octant, the body has to be inserted and set this octant as current node.

                                std::size_t octantID;
                                double parentMin_x = MIN_X[currentNode];
                                double parentMin_y = MIN_Y[currentNode];
                                double parentMin_z = MIN_Z[currentNode];
                                double parentEdgeLength = EDGE_LENGTHS[currentNode];

                                bool upperPart = POS_Y[i] > parentMin_y + (parentEdgeLength / 2);
                                bool rightPart = POS_X[i] > parentMin_x + (parentEdgeLength / 2);
                                bool backPart = POS_Z[i] < parentMin_z + (parentEdgeLength / 2);

                                if (!upperPart && !rightPart && !backPart) {
                                    octantID = LOWER_SW[currentNode];
                                } else if (!upperPart && !rightPart && backPart) {
                                    octantID = LOWER_NW[currentNode];
                                } else if (!upperPart && rightPart && !backPart) {
                                    octantID = LOWER_SE[currentNode];
                                } else if (!upperPart && rightPart && backPart) {
                                    octantID = LOWER_NE[currentNode];
                                } else if (upperPart && !rightPart && !backPart) {
                                    octantID = UPPER_SW[currentNode];
                                } else if (upperPart && !rightPart && backPart) {
                                    octantID = UPPER_NW[currentNode];
                                } else if (upperPart && rightPart && !backPart) {
                                    octantID = UPPER_SE[currentNode];
                                } else if (upperPart && rightPart && backPart) {
                                    octantID = UPPER_NE[currentNode];
                                }


//                                currentDepth += 1;

                                // update sum masses and center of mass of this node, since the current body will be inserted in one of the children
                                SUM_MASSES[currentNode] += MASSES[i];
                                CENTER_OF_MASS_X[currentNode] += POS_X[i] * MASSES[i];
                                CENTER_OF_MASS_Y[currentNode] += POS_Y[i] * MASSES[i];
                                CENTER_OF_MASS_Z[currentNode] += POS_Z[i] * MASSES[i];

                                currentNode = octantID;
                                currentDepth += 1;
                            }
                            nd_item.mem_fence(access::fence_space::global_and_local);
                            atomicNodeIsLockedAccessor.fetch_sub(1, memory_order::acq_rel, memory_scope::device);
//                            atomic_fence(memory_order::acq_rel,memory_scope::device);
                        }
                    }
                }
            }
        });
    }).wait();

//    host_accessor<double> testacc(sumOfMasses);
//    std::cout << testacc[0] << std::endl;

    // set the maximum tree depth
    host_accessor maxTreeDepthAccessor(maxTreeDepths);
    maxTreeDepth = maxTreeDepthAccessor[0];

    auto end = std::chrono::steady_clock::now();
    std::cout << "---------------------------------------------------------- "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;


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


//    std::cout << "max DEPTH: --------------- " << maxTreeDepth << std::endl;

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

//    host_accessor<std::size_t > TestHOST(testBuff);
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
