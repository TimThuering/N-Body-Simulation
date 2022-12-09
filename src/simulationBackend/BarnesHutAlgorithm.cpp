#include "BarnesHutAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>
#include "list"
#include "stack"
#include <chrono>

using namespace sycl;

BarnesHutAlgorithm::BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth,
                                       std::string &outputDirectory, std::size_t numberOfBodies)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory, numberOfBodies) {
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
    buildOctree(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z, masses);

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
        buildOctree(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z,
                    masses);

        computeAccelerations(queue, masses, intermediatePosition_x, intermediatePosition_y,
                             intermediatePosition_z,
                             acceleration_x, acceleration_y, acceleration_z);
        auto end = std::chrono::steady_clock::now();

        std::cout << "Time of step:  " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
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

    upper_NW.clear();
    upper_NE.clear();
    upper_SW.clear();
    upper_SE.clear();

    lower_NW.clear();
    lower_NE.clear();
    lower_SW.clear();
    lower_SE.clear();

    edgeLengths.clear();

    min_x_values.clear();
    min_y_values.clear();
    min_z_values.clear();

    bodyOfNode.clear();

    sumMasses.clear();

    centerOfMass_x.clear();
    centerOfMass_y.clear();
    centerOfMass_z.clear();
}

void
BarnesHutAlgorithm::buildOctree(queue &queue, buffer<double> &current_positions_x, buffer<double> &current_positions_y,
                                buffer<double> &current_positions_z, buffer<double> &masses) {

    host_accessor<double> POS_X(current_positions_x);
    host_accessor<double> POS_Y(current_positions_y);
    host_accessor<double> POS_Z(current_positions_z);

    host_accessor<double> MASSES(masses);

    std::size_t nextFreeNodeID = 0;
    std::size_t currentBody = 0;

    maxTreeDepth = 0;

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

    // initialize sum Masses and centerOfMass vectors
    sumMasses.push_back(MASSES[currentBody]);

    centerOfMass_x.push_back(POS_X[currentBody] * MASSES[currentBody]);
    centerOfMass_y.push_back(POS_Y[currentBody] * MASSES[currentBody]);
    centerOfMass_z.push_back(POS_Z[currentBody] * MASSES[currentBody]);

    nextFreeNodeID += 1;


    for (std::size_t i = 1; i < numberOfBodies; ++i) {
        std::size_t currentDepth = 0;
        std::size_t currentNode = 0;
        bool nodeInserted = false;
        while (!nodeInserted) {
            if (upper_NW[currentNode] == 0) {
                // the current node is a leaf node
                if (bodyOfNode[currentNode] == numberOfBodies) {
                    // the leaf node is empty --> Insert the current body into the current node and set the flag to continue with next body
                    bodyOfNode[currentNode] = i;

                    // update sum masses and center of mass
                    sumMasses[currentNode] += MASSES[i];
                    centerOfMass_x[currentNode] += POS_X[i] * MASSES[i];
                    centerOfMass_y[currentNode] += POS_Y[i] * MASSES[i];
                    centerOfMass_z[currentNode] += POS_Z[i] * MASSES[i];

                    if (currentDepth > maxTreeDepth) {
                        maxTreeDepth = currentDepth;
                    }
                    nodeInserted = true;
                } else {
                    // the leaf node already contains a body --> split the node and insert old body
                    std::size_t bodyIDinNode = bodyOfNode[currentNode];
                    splitNode(currentNode, nextFreeNodeID);
                    nextFreeNodeID += 8;

                    std::size_t octantID = getOctantContainingBody(POS_X[bodyIDinNode],
                                                                   POS_Y[bodyIDinNode],
                                                                   POS_Z[bodyIDinNode], currentNode);
                    // insert the old body into the new octant it belongs to and remove it from the parent node
                    bodyOfNode[octantID] = bodyIDinNode;
                    bodyOfNode[currentNode] = numberOfBodies;

                    sumMasses[octantID] += MASSES[bodyIDinNode];
                    centerOfMass_x[octantID] += POS_X[bodyIDinNode] * MASSES[bodyIDinNode];
                    centerOfMass_y[octantID] += POS_Y[bodyIDinNode] * MASSES[bodyIDinNode];
                    centerOfMass_z[octantID] += POS_Z[bodyIDinNode] * MASSES[bodyIDinNode];

                }
            } else {
                // the current node is not a leaf node, i.e. it has 8 children
                // --> determine the octant, the body has to be inserted and set this octant as current node.

                // update sum masses and center of mass of this node, since the current body will be inserted in one of the children
                sumMasses[currentNode] += MASSES[i];
                centerOfMass_x[currentNode] += POS_X[i] * MASSES[i];
                centerOfMass_y[currentNode] += POS_Y[i] * MASSES[i];
                centerOfMass_z[currentNode] += POS_Z[i] * MASSES[i];

                std::size_t octantID = getOctantContainingBody(POS_X[i], POS_Y[i],
                                                               POS_Z[i], currentNode);

                currentNode = octantID;
                currentDepth += 1;

            }

        }
    }
}

void BarnesHutAlgorithm::buildOctreeParallel(queue &queue, buffer<double> &current_positions_x,
                                             buffer<double> &current_positions_y, buffer<double> &current_positions_z,
                                             buffer<double> &masses) {


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

void BarnesHutAlgorithm::splitNode(std::size_t nodeID, std::size_t firstIndex) {

    double childEdgeLength = edgeLengths[nodeID] / 2;

    double parent_min_x = min_x_values[nodeID];
    double parent_min_y = min_y_values[nodeID];
    double parent_min_z = min_z_values[nodeID];


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
    min_x_values.push_back(parent_min_x);
    min_y_values.push_back(parent_min_y + childEdgeLength);
    min_z_values.push_back(parent_min_z);

    // min x,y,z values of the upperNE child node
    min_x_values.push_back(parent_min_x + childEdgeLength);
    min_y_values.push_back(parent_min_y + childEdgeLength);
    min_z_values.push_back(parent_min_z);

    // min x,y,z values of the upperSW child node
    min_x_values.push_back(parent_min_x);
    min_y_values.push_back(parent_min_y + childEdgeLength);
    min_z_values.push_back(parent_min_z + childEdgeLength);

    // min x,y,z values of the upperSE child node
    min_x_values.push_back(parent_min_x + childEdgeLength);
    min_y_values.push_back(parent_min_y + childEdgeLength);
    min_z_values.push_back(parent_min_z + childEdgeLength);

    // min x,y,z values of the lowerNW child node
    min_x_values.push_back(parent_min_x);
    min_y_values.push_back(parent_min_y);
    min_z_values.push_back(parent_min_z);

    // min x,y,z values of the lowerNE child node
    min_x_values.push_back(parent_min_x + childEdgeLength);
    min_y_values.push_back(parent_min_y);
    min_z_values.push_back(parent_min_z);

    // min x,y,z values of the lowerSW child node
    min_x_values.push_back(parent_min_x);
    min_y_values.push_back(parent_min_y);
    min_z_values.push_back(parent_min_z + childEdgeLength);

    // min x,y,z values of the lowerSE child node
    min_x_values.push_back(parent_min_x + childEdgeLength);
    min_y_values.push_back(parent_min_y);
    min_z_values.push_back(parent_min_z + childEdgeLength);

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

        centerOfMass_x.push_back(0);
        centerOfMass_y.push_back(0);
        centerOfMass_z.push_back(0);

        sumMasses.push_back(0);
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

void BarnesHutAlgorithm::computeAccelerations(queue &queue, buffer<double> &masses,
                                              buffer<double> &currentPositions_x,
                                              buffer<double> &currentPositions_y,
                                              buffer<double> &currentPositions_z,
                                              buffer<double> &acceleration_x, buffer<double> &acceleration_y,
                                              buffer<double> &acceleration_z) {

    auto begin = std::chrono::steady_clock::now();

    std::size_t stackSize = (8 * maxTreeDepth); // determine the stack size for each work item


    std::vector<std::size_t> nodesOnStack_vec(stackSize * numberOfBodies, bodyOfNode.size());

//    std::vector<std::size_t> test(numberOfBodies);
//    buffer<std::size_t> testBuff = test;

    buffer<std::size_t> nodesOnStack = nodesOnStack_vec;

    buffer<double> currentEdgeLengths = edgeLengths;

    buffer<double> sumOfMasses = sumMasses;

    buffer<double> massCenters_x = centerOfMass_x;
    buffer<double> massCenters_y = centerOfMass_y;
    buffer<double> massCenters_z = centerOfMass_z;

    buffer<std::size_t> upper_NW_buffer = upper_NW;
    buffer<std::size_t> upper_NE_buffer = upper_NE;
    buffer<std::size_t> upper_SW_buffer = upper_SW;
    buffer<std::size_t> upper_SE_buffer = upper_SE;
    buffer<std::size_t> lower_NW_buffer = lower_NW;
    buffer<std::size_t> lower_NE_buffer = lower_NE;
    buffer<std::size_t> lower_SW_buffer = lower_SW;
    buffer<std::size_t> lower_SE_buffer = lower_SE;

    buffer<std::size_t> bodyOfNode_buffer = bodyOfNode;

    buffer<double> min_x_values_buffer = min_x_values;
    buffer<double> min_y_values_buffer = min_y_values;
    buffer<double> min_z_values_buffer = min_z_values;

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

        accessor<double> EDGE_LENGTHS(currentEdgeLengths, h);
        accessor<double> CENTER_OF_MASS_X(massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(massCenters_z, h);

        accessor<std::size_t> UPPER_NW(upper_NW_buffer, h);
        accessor<std::size_t> UPPER_NE(upper_NE_buffer, h);
        accessor<std::size_t> UPPER_SW(upper_SW_buffer, h);
        accessor<std::size_t> UPPER_SE(upper_SE_buffer, h);
        accessor<std::size_t> LOWER_NW(lower_NW_buffer, h);
        accessor<std::size_t> LOWER_NE(lower_NE_buffer, h);
        accessor<std::size_t> LOWER_SW(lower_SW_buffer, h);
        accessor<std::size_t> LOWER_SE(lower_SE_buffer, h);

//        accessor<std::size_t> TestACC(testBuff, h);

        accessor<std::size_t> BODY_OF_NODE(bodyOfNode_buffer, h);

        accessor<double> MIN_X(min_x_values_buffer, h);
        accessor<double> MIN_Y(min_y_values_buffer, h);
        accessor<double> MIN_Z(min_z_values_buffer, h);

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
