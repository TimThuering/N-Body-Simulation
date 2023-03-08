#include "BarnesHutAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

BarnesHutAlgorithm::BarnesHutAlgorithm(double dt, double tEnd, double visualizationStepWidth,
                                       std::string &outputDirectory)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory),
          nodesOnStack_vec(configuration::barnes_hut_algorithm::stackSize * configuration::numberOfBodies,
                           octree.bodyOfNode.size()),
          nodesOnStack(nodesOnStack_vec.data(), nodesOnStack_vec.size()) {
    this->description = "Barnes-Hut Algorithm";
}


void BarnesHutAlgorithm::startSimulation(const SimulationData &simulationData) {
    // Vectors and their corresponding buffers for the acceleration values of each body induced by the gravitational
    // force from all other bodies in the dataset. The index corresponds the body id.
    std::vector<double> acc_x(configuration::numberOfBodies);
    std::vector<double> acc_y(configuration::numberOfBodies);
    std::vector<double> acc_z(configuration::numberOfBodies);

    buffer<double> acceleration_x(acc_x.data(), acc_x.size());
    buffer<double> acceleration_y(acc_y.data(), acc_y.size());
    buffer<double> acceleration_z(acc_z.data(), acc_z.size());

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

    buffer<double> intermediatePosition_x(intermediatePosition_x_vec.data(), intermediatePosition_x_vec.size());
    buffer<double> intermediatePosition_y(intermediatePosition_y_vec.data(), intermediatePosition_y_vec.size());
    buffer<double> intermediatePosition_z(intermediatePosition_z_vec.data(), intermediatePosition_z_vec.size());

    buffer<double> intermediateVelocity_x(intermediateVelocity_x_vec.data(), intermediateVelocity_x_vec.size());
    buffer<double> intermediateVelocity_y(intermediateVelocity_y_vec.data(), intermediateVelocity_y_vec.size());
    buffer<double> intermediateVelocity_z(intermediateVelocity_z_vec.data(), intermediateVelocity_z_vec.size());

    // SYCL queue for computation tasks
    queue queue;

    if (configuration::use_GPUs) {
        queue = sycl::queue(sycl::gpu_selector_v);

    } else {
        queue = sycl::queue(sycl::cpu_selector_v);
    }

    // vector containing all the masses of the bodies
    buffer<double> masses(simulationData.mass.data(), simulationData.mass.size());

    // current time of the simulation
    double time = 0.0;
    double timeSinceLastVisualization = 0.0; // used to determine the time steps that should be visualized

    // current visualization step of the simulation
    d_type::int_t currentStep = 0;

    // configure the timer
    std::string device = queue.get_device().get_info<info::device::name>();
    timer.setProperties(description, configuration::numberOfBodies, device);
    timer.addTimingSequence("Total Time");
    timer.addTimingSequence("Octree creation");
    timer.addTimingSequence("Acceleration Kernel Time");
    timer.addTimingSequence("AABB creation");
    timer.addTimingSequence("Compute center of mass");

#ifdef OCTREE_TOP_DOWN_SYNC
    timer.addTimingSequence("Build octree");
#else
    timer.addTimingSequence("Build octree to level");
    timer.addTimingSequence("Prepare subtrees");
    timer.addTimingSequence("Sort bodies for subtrees");
    timer.addTimingSequence("Build subtrees");
#endif

    if (configuration::barnes_hut_algorithm::sortBodies) {
        timer.addTimingSequence("Sort bodies");
    }

    // start of the simulation:
    // computations for initial state: all values get stored for the output

    auto begin1 = std::chrono::steady_clock::now();
    octree.buildOctree(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z, masses, timer);
    auto endTreeCreation1 = std::chrono::steady_clock::now();


    // compute initial accelerations
    auto beginAccelerationKernel1 = std::chrono::steady_clock::now();
    computeAccelerations(queue, masses, intermediatePosition_x, intermediatePosition_y,
                         intermediatePosition_z,
                         acceleration_x, acceleration_y, acceleration_z);
    auto end1 = std::chrono::steady_clock::now();
    timer.addTimeToSequence("Octree creation",
                            std::chrono::duration<double, std::milli>(endTreeCreation1 - begin1).count());
    timer.addTimeToSequence("Acceleration Kernel Time",
                            std::chrono::duration<double, std::milli>(end1 - beginAccelerationKernel1).count());
    timer.addTimeToSequence("Total Time", std::chrono::duration<double, std::milli>(end1 - begin1).count());


    if (configuration::compute_energy) {
        // compute energy of the initial step.
        computeEnergy(queue, masses, currentStep, intermediatePosition_x, intermediatePosition_y,
                      intermediatePosition_z, intermediateVelocity_x, intermediateVelocity_y, intermediateVelocity_z);
    }


    // store norm of all accelerations
    storeAccelerations(currentStep, acceleration_x, acceleration_y, acceleration_z);
    std::cout << "Step " << currentStep << std::endl;

    // continue with next simulation step
    time += dt;
    timeSinceLastVisualization += dt;
    currentStep += 1;

    // vectors for intermediate values during the time integration
    std::vector<double> vx_k1_2_vec(configuration::numberOfBodies);
    std::vector<double> vy_k1_2_vec(configuration::numberOfBodies);
    std::vector<double> vz_k1_2_vec(configuration::numberOfBodies);

    buffer<double> vx_k1_2(vx_k1_2_vec.data(), vx_k1_2_vec.size());
    buffer<double> vy_k1_2(vy_k1_2_vec.data(), vy_k1_2_vec.size());
    buffer<double> vz_k1_2(vz_k1_2_vec.data(), vz_k1_2_vec.size());


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

            h.parallel_for(sycl::range<1>(configuration::numberOfBodies), [=](auto &i) {

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

            for (d_type::int_t i = 0; i < configuration::numberOfBodies; ++i) {
                positions_x[currentStep].push_back(INTERMEDIATE_P_X[i]);
                positions_y[currentStep].push_back(INTERMEDIATE_P_Y[i]);
                positions_z[currentStep].push_back(INTERMEDIATE_P_Z[i]);
            }
        }

        auto begin = std::chrono::steady_clock::now();
        octree.buildOctree(queue, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z,
                           masses, timer);
        auto endTreeCreation = std::chrono::steady_clock::now();

        auto beginAccelerationKernel = std::chrono::steady_clock::now();
        computeAccelerations(queue, masses, intermediatePosition_x, intermediatePosition_y,
                             intermediatePosition_z,
                             acceleration_x, acceleration_y, acceleration_z);
        auto end = std::chrono::steady_clock::now();

        timer.addTimeToSequence("Octree creation",
                                std::chrono::duration<double, std::milli>(endTreeCreation - begin).count());
        timer.addTimeToSequence("Acceleration Kernel Time",
                                std::chrono::duration<double, std::milli>(end - beginAccelerationKernel).count());
        timer.addTimeToSequence("Total Time", std::chrono::duration<double, std::milli>(end - begin).count());


        std::cout << "Time of step:  " << std::chrono::duration<double, std::milli>(end - begin).count()
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

            h.parallel_for(sycl::range<1>(configuration::numberOfBodies), [=](auto &i) {
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
                for (d_type::int_t i = 0; i < configuration::numberOfBodies; ++i) {
                    velocities_x[currentStep].push_back(INTERMEDIATE_V_X[i]);
                    velocities_y[currentStep].push_back(INTERMEDIATE_V_Y[i]);
                    velocities_z[currentStep].push_back(INTERMEDIATE_V_Z[i]);
                }
            }

            if (configuration::compute_energy) {
                // compute and store energy values of the system in the current time step
                computeEnergy(queue, masses, currentStep, intermediatePosition_x, intermediatePosition_y,
                              intermediatePosition_z, intermediateVelocity_x, intermediateVelocity_y,
                              intermediateVelocity_z);
            }

            // reset time for next visualization time step
            currentStep += 1;
            timeSinceLastVisualization = 0.0;
        }

        // continue with next simulation step
        time += dt;
        timeSinceLastVisualization += dt;
    }
}

void BarnesHutAlgorithm::computeAccelerations(queue &queue, buffer<double> &masses,
                                              buffer<double> &currentPositions_x,
                                              buffer<double> &currentPositions_y,
                                              buffer<double> &currentPositions_z,
                                              buffer<double> &acceleration_x, buffer<double> &acceleration_y,
                                              buffer<double> &acceleration_z) {
    auto begin = std::chrono::steady_clock::now();

    double epsilon_2 = configuration::epsilon2;
    d_type::int_t N = configuration::numberOfBodies;
    double G = this->G;
    double THETA = configuration::barnes_hut_algorithm::theta;
    d_type::int_t storageSize = configuration::barnes_hut_algorithm::storageSizeParameter;
    d_type::int_t stack_size = configuration::barnes_hut_algorithm::stackSize;
    bool bodiesSorted = configuration::barnes_hut_algorithm::sortBodies;

    int workGroupSize = configuration::barnes_hut_algorithm::workGroupSize;
    d_type::int_t padding = workGroupSize - (configuration::numberOfBodies % workGroupSize);


    queue.submit([&](handler &h) {
        accessor<double> POS_X(currentPositions_x, h);
        accessor<double> POS_Y(currentPositions_y, h);
        accessor<double> POS_Z(currentPositions_z, h);
        accessor<double> SUM_MASSES(octree.sumOfMasses, h);
        accessor<double> ACC_X(acceleration_x, h);
        accessor<double> ACC_Y(acceleration_y, h);
        accessor<double> ACC_Z(acceleration_z, h);
        accessor<double> EDGE_LENGTHS(octree.edgeLengths, h);
        accessor<double> CENTER_OF_MASS_X(octree.massCenters_x, h);
        accessor<double> CENTER_OF_MASS_Y(octree.massCenters_y, h);
        accessor<double> CENTER_OF_MASS_Z(octree.massCenters_z, h);
        accessor<d_type::int_t> OCTANTS(octree.octants, h);
        accessor<d_type::int_t> BODY_OF_NODE(octree.bodyOfNode, h);
        accessor<d_type::int_t> NODES_ON_STACK(nodesOnStack, h);
        accessor<d_type::int_t> SORTED_BODIES(octree.sortedBodiesInOrder, h);

        auto kernelRange = nd_range<1>(range<1>(configuration::numberOfBodies + padding), range<1>(workGroupSize));

        h.parallel_for(kernelRange, [=](auto &nd_item) {
            d_type::int_t id = nd_item.get_global_id();

            if (id < N) {
                // i > N is not valid since it is part of the padding.
                double acc_x = 0;
                double acc_y = 0;
                double acc_z = 0;

                // determine current body ID of work-item
                d_type::int_t i;
                if (bodiesSorted) {
                    i = SORTED_BODIES[id];
                } else {
                    i = id;
                }

                double pos_x_i = POS_X[i];
                double pos_y_i = POS_Y[i];
                double pos_z_i = POS_Z[i];

                d_type::int_t currentStackIndex = (stack_size * i) + 1; // start index for the stack of current work item.
                NODES_ON_STACK[currentStackIndex] = 0; // push the root node on the stack

                while (currentStackIndex != (stack_size * i)) { // while stack not empty

                    d_type::int_t current_Node = NODES_ON_STACK[currentStackIndex]; // get node from stack
                    currentStackIndex -= 1;

                    if (SUM_MASSES[current_Node] != 0 && BODY_OF_NODE[current_Node] != i) {
                        // the node is not empty and does not contain the current body.
                        double d_x = (CENTER_OF_MASS_X[current_Node] / SUM_MASSES[current_Node]) - pos_x_i;
                        double d_y = (CENTER_OF_MASS_Y[current_Node] / SUM_MASSES[current_Node]) - pos_y_i;
                        double d_z = (CENTER_OF_MASS_Z[current_Node] / SUM_MASSES[current_Node]) - pos_z_i;

                        double d = sycl::rsqrt(d_x * d_x + d_y * d_y + d_z * d_z);

                        double currentTheta = EDGE_LENGTHS[current_Node] * d;

                        if (((currentTheta < THETA) || BODY_OF_NODE[current_Node] != N)) {
                            // center of mass of the node can be used to compute the acceleration
                            double denominator = (d_x * d_x) + (d_y * d_y) + (d_z * d_z) + epsilon_2;
                            denominator = denominator * denominator * denominator;
                            denominator = sycl::rsqrt(denominator);

                            acc_x += SUM_MASSES[current_Node] * (d_x * denominator);
                            acc_y += SUM_MASSES[current_Node] * (d_y * denominator);
                            acc_z += SUM_MASSES[current_Node] * (d_z * denominator);
                        } else {
                            // center of mass of the node can not be used --> Push all children on the stack and continue
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[5 * storageSize + current_Node];
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[7 * storageSize + current_Node];
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[4 * storageSize + current_Node];
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[6 * storageSize + current_Node];
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[1 * storageSize + current_Node];
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[3 * storageSize + current_Node];
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[0 + current_Node];
                            currentStackIndex += 1;
                            NODES_ON_STACK[currentStackIndex] = OCTANTS[2 * storageSize + current_Node];
                        }
                    }
                }
                ACC_X[i] = acc_x * G;
                ACC_Y[i] = acc_y * G;
                ACC_Z[i] = acc_z * G;
            }
        });
    }).wait();


    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double, std::milli>(end - begin).count() << std::endl;

}
