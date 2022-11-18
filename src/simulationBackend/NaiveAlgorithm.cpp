#include "NaiveAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>
#include <chrono>

using namespace sycl;

NaiveAlgorithm::NaiveAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory,
                               std::size_t numberOfBodies)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory, numberOfBodies) {
    this->description = "Naive Algorithm";
}


void NaiveAlgorithm::startSimulation(const SimulationData &simulationData) {
    auto begin = std::chrono::steady_clock::now();
    // Vectors for the acceleration values of each body induced by the gravitational force from all other bodies in
    // the dataset. The index corresponds the body id.
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
    buffer<double> intermediatePosition_x = simulationData.positions_x;
    buffer<double> intermediatePosition_y = simulationData.positions_y;
    buffer<double> intermediatePosition_z = simulationData.positions_z;

    buffer<double> intermediateVelocity_x = simulationData.velocities_x;
    buffer<double> intermediateVelocity_y = simulationData.velocities_y;
    buffer<double> intermediateVelocity_z = simulationData.velocities_z;



    // SYCL queue for computation tasks
    queue queue;

    // Buffer that stores all the masses of the bodies.
    buffer masses = simulationData.mass;

    // current time of the simulation
    double time = 0.0;
    double timeSinceLastVisualization = 0.0;

    // current visualization step of the simulation
    std::size_t currentStep = 0;

    // computations of initial state: all values get stored for the output

    // compute initial accelerations
    computeAccelerationsOptimized2(queue, masses, intermediatePosition_x, intermediatePosition_y,
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



    // start of simulation
    while (time < t_end) {

        bool visualizeCurrentStep = (std::abs(timeSinceLastVisualization - visualizationStepWidth) < 0.000001);

        double delta_t = dt;

        // leapfrog integration part 1 updates the position
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

        // store the current body positions for visualization
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


        // update the acceleration values, only depends on new position of bodies
        computeAccelerationsOptimized2(queue, masses, intermediatePosition_x, intermediatePosition_y,
                                       intermediatePosition_z,
                                       acceleration_x, acceleration_y, acceleration_z);


        // leapfrog integration part 2, update velocities based on the new computed acceleration
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

    auto end = std::chrono::steady_clock::now();
    std::cout << "Total time:  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;

}

void NaiveAlgorithm::adjustVelocities(const SimulationData &simulationData) {
    double sumMasses = 0;
    double sumMassesVelocity_x = 0;
    double sumMassesVelocity_y = 0;
    double sumMassesVelocity_z = 0;

    for (int i = 0; i < numberOfBodies; ++i) {
        sumMasses += simulationData.mass[i];
        sumMassesVelocity_x += simulationData.mass[i] * simulationData.velocities_x[i];
        sumMassesVelocity_y += simulationData.mass[i] * simulationData.velocities_y[i];
        sumMassesVelocity_z += simulationData.mass[i] * simulationData.velocities_z[i];
    }

    double ui_x = sumMassesVelocity_x / sumMasses;
    double ui_y = sumMassesVelocity_y / sumMasses;
    double ui_z = sumMassesVelocity_z / sumMasses;


    for (int i = 0; i < numberOfBodies; ++i) {
        velocities_x[0][i] -= ui_x;
        velocities_y[0][i] -= ui_y;
        velocities_z[0][i] -= ui_z;
    }
}

void NaiveAlgorithm::computeEnergy(queue &queue, buffer<double> &masses, std::size_t currentStep,
                                   buffer<double> &currentPositions_x,
                                   buffer<double> &currentPositions_y, buffer<double>
                                   &currentPositions_z,
                                   buffer<double> &velocities_x, buffer<double>
                                   &velocities_y,
                                   buffer<double> &velocities_z) {
    double E_kin_result = 0;
    double E_pot_result = 0;
    double E_total_result;


    std::vector<double> E_kin(numberOfBodies);
    std::vector<double> E_pot(numberOfBodies);

    buffer<double> E_kin_values = E_kin;
    buffer<double> E_pot_values = E_pot;

    double G = this->G;

    queue.submit([&](handler &h) {

        accessor<double> V_X(velocities_x, h);
        accessor<double> V_Y(velocities_y, h);
        accessor<double> V_Z(velocities_z, h);

        accessor<double> P_X(currentPositions_x, h);
        accessor<double> P_Y(currentPositions_y, h);
        accessor<double> P_Z(currentPositions_z, h);

        accessor<double> E_KIN(E_kin_values, h);
        accessor<double> E_POT(E_pot_values, h);

        accessor<double> M(masses, h);


        h.parallel_for(numberOfBodies, [=](auto &j) {
            double v = V_X[j] * V_X[j] +
                       V_Y[j] * V_Y[j] +
                       V_Z[j] * V_Z[j];
            E_KIN[j] = 0.5 * M[j] * v;
            E_POT[j] = 0;

            for (int i = 0; i < j; ++i) {
                double r_x = P_X[j] - P_X[i];
                double r_y = P_Y[j] - P_Y[i];
                double r_z = P_Z[j] - P_Z[i];

                double r = sycl::sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

                E_POT[j] += G * M[i] * M[j] / r;
            }
        });
    }).wait();


    host_accessor<double> E_KIN(E_kin_values);
    host_accessor<double> E_POT(E_pot_values);

    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        E_kin_result += E_KIN[i];
        E_pot_result += E_POT[i];
    }


    E_pot_result *= -1;
    // total energy
    E_total_result = E_kin_result + E_pot_result;

    totalEnergy[currentStep] = E_total_result;
    potentialEnergy[currentStep] = E_pot_result;
    kineticEnergy[currentStep] = E_kin_result;
    virialEquilibrium[currentStep] = (2.0 * E_kin_result) / std::abs(E_pot_result);
}

void NaiveAlgorithm::computeAccelerations(queue &queue, buffer<double> &masses, std::vector<double> &currentPositions_x,
                                          std::vector<double> &currentPositions_y,
                                          std::vector<double> &currentPositions_z,
                                          std::vector<double> &acceleration_x, std::vector<double> &acceleration_y,
                                          std::vector<double> &acceleration_z) {
    auto begin = std::chrono::steady_clock::now();
    double epsilon_2;
    epsilon_2 = pow(10, -22);

    buffer<double> pos_x = currentPositions_x;
    buffer<double> pos_y = currentPositions_y;
    buffer<double> pos_z = currentPositions_z;

    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        buffer<double> accelerations_x{range{numberOfBodies}};
        buffer<double> accelerations_y{range{numberOfBodies}};
        buffer<double> accelerations_z{range{numberOfBodies}};

        queue.submit([&](handler &h) {
            accessor<double> POS_X(pos_x, h);
            accessor<double> POS_Y(pos_y, h);
            accessor<double> POS_Z(pos_z, h);

            accessor<double> MASSES(masses, h);

            accessor<double> ACC_X(accelerations_x, h);
            accessor<double> ACC_Y(accelerations_y, h);
            accessor<double> ACC_Z(accelerations_z, h);

            // device code
            h.parallel_for(numberOfBodies, [=](auto &j) {
                double r_x = POS_X[j] - POS_X[i];
                double r_y = POS_Y[j] - POS_Y[i];
                double r_z = POS_Z[j] - POS_Z[i];

                double denominator = sycl::sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z) + epsilon_2);
                denominator = denominator * denominator * denominator;

                ACC_X[j] = MASSES[j] * (r_x / denominator);
                ACC_Y[j] = MASSES[j] * (r_y / denominator);
                ACC_Z[j] = MASSES[j] * (r_z / denominator);
            });
        }).wait();


        host_accessor<double> ACC_X(accelerations_x);
        host_accessor<double> ACC_Y(accelerations_y);
        host_accessor<double> ACC_Z(accelerations_z);

        double acc_x = 0;
        double acc_y = 0;
        double acc_z = 0;

        // sum up accelerations
        for (std::size_t idx = 0; idx < numberOfBodies; ++idx) {
            if (idx != i) {
                acc_x += ACC_X[idx];
                acc_y += ACC_Y[idx];
                acc_z += ACC_Z[idx];
            }
        }

        acceleration_x.at(i) = G * acc_x;
        acceleration_y.at(i) = G * acc_y;
        acceleration_z.at(i) = G * acc_z;
    }

    auto end = std::chrono::steady_clock::now();

    std::cout << "Accelerations:  " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;


}

void NaiveAlgorithm::computeAccelerationsOptimized(queue &queue, buffer<double> &masses,
                                                   buffer<double> &currentPositions_x,
                                                   buffer<double> &currentPositions_y,
                                                   buffer<double> &currentPositions_z,
                                                   buffer<double> &acceleration_x,
                                                   buffer<double> &acceleration_y,
                                                   buffer<double> &acceleration_z) {

    auto begin = std::chrono::steady_clock::now();
    double epsilon_2 = pow(10, -22);

    queue.submit([&](handler &h) {

        accessor<double> POS_X(currentPositions_x, h);
        accessor<double> POS_Y(currentPositions_y, h);
        accessor<double> POS_Z(currentPositions_z, h);

        accessor<double> MASSES(masses, h);

        accessor<double> ACC_X(acceleration_x, h);
        accessor<double> ACC_Y(acceleration_y, h);
        accessor<double> ACC_Z(acceleration_z, h);

        std::size_t N = numberOfBodies;
        double G = this->G;

        // device code
        h.parallel_for(numberOfBodies, [=](auto &i) {
            ACC_X[i] = 0;
            ACC_Y[i] = 0;
            ACC_Z[i] = 0;

            for (std::size_t j = 0; j < N; ++j) {

                double r_x = POS_X[j] - POS_X[i];
                double r_y = POS_Y[j] - POS_Y[i];
                double r_z = POS_Z[j] - POS_Z[i];

                double denominator = sycl::sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z) + epsilon_2);
                denominator = denominator * denominator * denominator;

                ACC_X[i] += MASSES[j] * (r_x / denominator);
                ACC_Y[i] += MASSES[j] * (r_y / denominator);
                ACC_Z[i] += MASSES[j] * (r_z / denominator);
            }

            ACC_X[i] *= G;
            ACC_Y[i] *= G;
            ACC_Z[i] *= G;
        });
    }).wait();
    auto end = std::chrono::steady_clock::now();

    std::cout << "Acceleration Kernel Time:  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;
}

void NaiveAlgorithm::computeAccelerationsOptimized2(queue &queue, buffer<double> &masses,
                                                    buffer<double> &currentPositions_x,
                                                    buffer<double> &currentPositions_y,
                                                    buffer<double> &currentPositions_z,
                                                    buffer<double> &acceleration_x,
                                                    buffer<double> &acceleration_y,
                                                    buffer<double> &acceleration_z) {

    auto begin = std::chrono::steady_clock::now();
    double epsilon_2 = pow(10, -22);

    queue.submit([&](handler &h) {

        accessor<double> POS_X(currentPositions_x, h);
        accessor<double> POS_Y(currentPositions_y, h);
        accessor<double> POS_Z(currentPositions_z, h);

        accessor<double> MASSES(masses, h);

        accessor<double> ACC_X(acceleration_x, h);
        accessor<double> ACC_Y(acceleration_y, h);
        accessor<double> ACC_Z(acceleration_z, h);

        std::size_t N = numberOfBodies;
        int tileSize = 64;
        std::size_t padding = tileSize - (numberOfBodies % tileSize);

        double G = this->G;

        local_accessor<double> LOCAL_MASSES(tileSize, h);
        local_accessor<double> LOCAL_POS_X(tileSize, h);
        local_accessor<double> LOCAL_POS_Y(tileSize, h);
        local_accessor<double> LOCAL_POS_Z(tileSize, h);

        // device code
        h.parallel_for(nd_range<1>(range<1>(numberOfBodies + padding), range<1>(tileSize)), [=](auto &nd_item) {
            int i = nd_item.get_global_id();
            if (i < N) {
                double acc_x = 0;
                double acc_y = 0;
                double acc_z = 0;

                double pos_x = POS_X[i];
                double pos_y = POS_Y[i];
                double pos_z = POS_Z[i];

                int local_id = nd_item.get_local_id();

                for (int k = 0; k < N; k += tileSize) {

                    if (local_id + k < N) {
                        LOCAL_MASSES[local_id] = MASSES[k + local_id];
                        LOCAL_POS_X[local_id] = POS_X[k + local_id];
                        LOCAL_POS_Y[local_id] = POS_Y[k + local_id];
                        LOCAL_POS_Z[local_id] = POS_Z[k + local_id];
                    }
                    nd_item.barrier();

                    for (int j = 0; j < tileSize && j + k < N; ++j) {
                        double r_x = LOCAL_POS_X[j] - pos_x;
                        double r_y = LOCAL_POS_Y[j] - pos_y;
                        double r_z = LOCAL_POS_Z[j] - pos_z;

                        double denominator = sycl::sqrt((r_x * r_x) + (r_y * r_y) + (r_z * r_z) + epsilon_2);
                        denominator = denominator * denominator * denominator;

                        acc_x += LOCAL_MASSES[j] * (r_x / denominator);
                        acc_y += LOCAL_MASSES[j] * (r_y / denominator);
                        acc_z += LOCAL_MASSES[j] * (r_z / denominator);
                    }
                    nd_item.barrier();
                }

                ACC_X[i] = acc_x * G;
                ACC_Y[i] = acc_y * G;
                ACC_Z[i] = acc_z * G;
            }
        });
    }).wait();
    auto end = std::chrono::steady_clock::now();

    std::cout << "Acceleration Kernel Time:  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << std::endl;
}

void NaiveAlgorithm::storeAccelerations(std::size_t currentStep, buffer<double> &acceleration_x,
                                        buffer<double> &acceleration_y, buffer<double> &acceleration_z) {

    host_accessor<double> ACC_X(acceleration_x);
    host_accessor<double> ACC_Y(acceleration_y);
    host_accessor<double> ACC_Z(acceleration_z);

    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        double accelerationNorm = ACC_X[i] * ACC_X[i] +
                                  ACC_Y[i] * ACC_Y[i] +
                                  ACC_Z[i] * ACC_Z[i];
        acceleration[currentStep].push_back(std::sqrt(accelerationNorm));
    }

}


