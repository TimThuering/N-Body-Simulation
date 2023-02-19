#include "NaiveAlgorithm.hpp"
#include "SimulationData.hpp"
#include "Configuration.hpp"
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

NaiveAlgorithm::NaiveAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory) {
    this->description = "Naive Algorithm";
}


void NaiveAlgorithm::startSimulation(const SimulationData &simulationData) {
    auto begin = std::chrono::steady_clock::now();
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

    // vector containing all the masses of the bodies
    std::vector<double> masses_vec = simulationData.mass;
    buffer<double> masses(simulationData.mass.data(), simulationData.mass.size());

    // current time of the simulation
    double time = 0.0;
    double timeSinceLastVisualization = 0.0; // used to determine the time steps that should be visualized

    // current visualization step of the simulation
    std::size_t currentStep = 0;

    // configure timer for time tracking
    std::string device = "GPU";
    timer.setProperties(description, configuration::numberOfBodies, device);
    timer.addTimingSequence("Acceleration Kernel Time");

    // start of the simulation:
    // computations for initial state: all values get stored for the output

    // compute initial accelerations
    auto beginAcc1 = std::chrono::steady_clock::now();
    computeAccelerationsGPU(queue, masses, intermediatePosition_x, intermediatePosition_y,
                            intermediatePosition_z,
                            acceleration_x, acceleration_y, acceleration_z);
    auto endAcc1 = std::chrono::steady_clock::now();
    timer.addTimeToSequence("Acceleration Kernel Time",
                            std::chrono::duration<double, std::milli>(endAcc1 - beginAcc1).count());
    std::cout << std::chrono::duration<double, std::milli>(endAcc1 - beginAcc1).count() << std::endl;

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

            for (std::size_t i = 0; i < configuration::numberOfBodies; ++i) {
                positions_x[currentStep].push_back(INTERMEDIATE_P_X[i]);
                positions_y[currentStep].push_back(INTERMEDIATE_P_Y[i]);
                positions_z[currentStep].push_back(INTERMEDIATE_P_Z[i]);
            }
        }


        auto beginAcc = std::chrono::steady_clock::now();
        // update the acceleration values, only depends on new position of bodies
        computeAccelerationsGPU(queue, masses, intermediatePosition_x, intermediatePosition_y,
                                intermediatePosition_z,
                                acceleration_x, acceleration_y, acceleration_z);
        auto endAcc = std::chrono::steady_clock::now();
        timer.addTimeToSequence("Acceleration Kernel Time",
                                std::chrono::duration<double, std::milli>(endAcc - beginAcc).count());



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
                for (std::size_t i = 0; i < configuration::numberOfBodies; ++i) {
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

    auto end = std::chrono::steady_clock::now();
    std::cout << "Total time:  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;

}

void NaiveAlgorithm::computeAccelerationsGPU(queue &queue, buffer<double> &masses,
                                             buffer<double> &currentPositions_x,
                                             buffer<double> &currentPositions_y,
                                             buffer<double> &currentPositions_z,
                                             buffer<double> &acceleration_x,
                                             buffer<double> &acceleration_y,
                                             buffer<double> &acceleration_z) {

    auto begin = std::chrono::steady_clock::now();
    double epsilon_2 = configuration::epsilon2;

    queue.submit([&](handler &h) {

        accessor<double> POS_X(currentPositions_x, h);
        accessor<double> POS_Y(currentPositions_y, h);
        accessor<double> POS_Z(currentPositions_z, h);

        accessor<double> MASSES(masses, h);

        accessor<double> ACC_X(acceleration_x, h);
        accessor<double> ACC_Y(acceleration_y, h);
        accessor<double> ACC_Z(acceleration_z, h);

        std::size_t N = configuration::numberOfBodies;
        int tileSize = configuration::naive_algorithm::tileSizeNaiveAlg;

        // global size of the nd_range kernel has to be divisible by the tile size (local size).
        // The purpose of the padding is that numberOfBodies + padding is divisible by the tile size.
        std::size_t padding = tileSize - (configuration::numberOfBodies % tileSize);

        double G = this->G;

        // Accessors for masses and positions in local memory
        local_accessor<double> LOCAL_MASSES(tileSize, h);
        local_accessor<double> LOCAL_POS_X(tileSize, h);
        local_accessor<double> LOCAL_POS_Y(tileSize, h);
        local_accessor<double> LOCAL_POS_Z(tileSize, h);
        
        // device code
        h.parallel_for(nd_range<1>(range<1>(configuration::numberOfBodies + padding), range<1>(tileSize)),
                       [=](auto &nd_item) {
                           int i = nd_item.get_global_id();
                           double pos_x;
                           double pos_y;
                           double pos_z;

                           double acc_x = 0;
                           double acc_y = 0;
                           double acc_z = 0;

                           if (i < N) { // i > N is not valid since it is part of the padding.
                               pos_x = POS_X[i];
                               pos_y = POS_Y[i];
                               pos_z = POS_Z[i];
                           }

                           int local_id = nd_item.get_local_id();

                           // iterate over all other bodies in blocks of tile size. After each tile, reload the all values from global into local memory.
                           for (int k = 0; k < N; k += tileSize) {

                               // if > N, we have already reached the last body and all other values would be part of the padding.
                               if (local_id + k < N) {
                                   LOCAL_MASSES[local_id] = MASSES[k + local_id];
                                   LOCAL_POS_X[local_id] = POS_X[k + local_id];
                                   LOCAL_POS_Y[local_id] = POS_Y[k + local_id];
                                   LOCAL_POS_Z[local_id] = POS_Z[k + local_id];
                               }

                               nd_item.barrier(); // start with computation only when all values are loaded into the local memory.

                               if (i < N) {
                                   for (int j = 0; j < tileSize && j + k < N; ++j) {
                                       double r_x = LOCAL_POS_X[j] - pos_x;
                                       double r_y = LOCAL_POS_Y[j] - pos_y;
                                       double r_z = LOCAL_POS_Z[j] - pos_z;

                                       double denominator =
                                               (r_x * r_x) + (r_y * r_y) + (r_z * r_z) + epsilon_2;
                                       denominator = denominator * denominator * denominator;
                                       denominator = sycl::rsqrt(denominator);

//                                       double denominator = sycl::sqrt(
//                                               (r_x * r_x) + (r_y * r_y) + (r_z * r_z) + epsilon_2);
//                                       denominator = denominator * denominator * denominator;
//
//                                       acc_x += LOCAL_MASSES[j] * (r_x / denominator);
//                                       acc_y += LOCAL_MASSES[j] * (r_y / denominator);
//                                       acc_z += LOCAL_MASSES[j] * (r_z / denominator);

                                       acc_x += LOCAL_MASSES[j] * (r_x * denominator);
                                       acc_y += LOCAL_MASSES[j] * (r_y * denominator);
                                       acc_z += LOCAL_MASSES[j] * (r_z * denominator);
                                   }
                               }
                               nd_item.barrier(); // continue with loading new values only when all computations of current tile are done
                           }

                           if (i < N) {
                               ACC_X[i] = acc_x * G;
                               ACC_Y[i] = acc_y * G;
                               ACC_Z[i] = acc_z * G;
                           }
                       });
    }).wait();
    auto end = std::chrono::steady_clock::now();

    std::cout << "Acceleration Kernel Time:  "
              << std::chrono::duration<double, std::milli>(end - begin).count()
              << std::endl;
}

void NaiveAlgorithm::computeAccelerationsCPU(queue &queue, buffer<double> &masses, buffer<double> &currentPositions_x,
                                             buffer<double> &currentPositions_y, buffer<double> &currentPositions_z,
                                             buffer<double> &acceleration_x, buffer<double> &acceleration_y,
                                             buffer<double> &acceleration_z) {

    auto begin = std::chrono::steady_clock::now();
    double epsilon_2 = configuration::epsilon2;
    double G = this->G;
    std::size_t N = configuration::numberOfBodies;

    queue.submit([&](handler &h) {

        accessor<double> POS_X(currentPositions_x, h);
        accessor<double> POS_Y(currentPositions_y, h);
        accessor<double> POS_Z(currentPositions_z, h);
        accessor<double> MASSES(masses, h);
        accessor<double> ACC_X(acceleration_x, h);
        accessor<double> ACC_Y(acceleration_y, h);
        accessor<double> ACC_Z(acceleration_z, h);

        // device code
        h.parallel_for(sycl::range<1>(configuration::numberOfBodies), [=](auto &i) {
            double acc_x = 0;
            double acc_y = 0;
            double acc_z = 0;

            for (std::size_t j = 0; j < N; ++j) {

                double r_x = POS_X[j] - POS_X[i];
                double r_y = POS_Y[j] - POS_Y[i];
                double r_z = POS_Z[j] - POS_Z[i];

                double denominator = (r_x * r_x) + (r_y * r_y) + (r_z * r_z) + epsilon_2;
                denominator = denominator * denominator * denominator;
                denominator = sycl::rsqrt(denominator);

                acc_x += MASSES[j] * (r_x * denominator);
                acc_y += MASSES[j] * (r_y * denominator);
                acc_z += MASSES[j] * (r_z * denominator);
            }

            ACC_X[i] = acc_x * G;
            ACC_Y[i] = acc_y * G;
            ACC_Z[i] = acc_z * G;

        });
    }).wait();
    auto end = std::chrono::steady_clock::now();

    std::cout << "Acceleration Kernel Time:  "
              << std::chrono::duration<double, std::milli>(end - begin).count()
              << std::endl;

}
