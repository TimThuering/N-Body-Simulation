#include <iostream>
#include "NaiveAlgorithm.hpp"
#include "SimulationData.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

NaiveAlgorithm::NaiveAlgorithm(double dt, double tEnd, double visualizationStepWidth, std::string &outputDirectory,
                               std::size_t numberOfBodies)
        : nBodyAlgorithm(dt, tEnd, visualizationStepWidth, outputDirectory, numberOfBodies) {
    this->description = "Naive Algorithm";
}


void NaiveAlgorithm::startSimulation(const SimulationData &simulationData) {
    // Vectors for the acceleration values of each body induced by the gravitational force from all other bodies in
    // the dataset. The index corresponds the body id.
    std::vector<double> acceleration_x(numberOfBodies);
    std::vector<double> acceleration_y(numberOfBodies);
    std::vector<double> acceleration_z(numberOfBodies);

    // set initial values of position and velocity to the values read directly form the dataset
    positions_x[0] = simulationData.positions_x;
    positions_y[0] = simulationData.positions_y;
    positions_z[0] = simulationData.positions_z;

    velocities_x[0] = simulationData.velocities_x;
    velocities_y[0] = simulationData.velocities_y;
    velocities_z[0] = simulationData.velocities_z;

    // vectors for intermediate position and velocity values;
    std::vector<double> intermediatePosition_x(numberOfBodies);
    std::vector<double> intermediatePosition_y(numberOfBodies);
    std::vector<double> intermediatePosition_z(numberOfBodies);

    std::vector<double> intermediateVelocity_x(numberOfBodies);
    std::vector<double> intermediateVelocity_y(numberOfBodies);
    std::vector<double> intermediateVelocity_z(numberOfBodies);

    // copy initial position and velocity values
    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        intermediatePosition_x.at(i) = positions_x[0][i];
        intermediatePosition_y.at(i) = positions_y[0][i];
        intermediatePosition_z.at(i) = positions_z[0][i];

        intermediateVelocity_x.at(i) = velocities_x[0][i];
        intermediateVelocity_y.at(i) = velocities_y[0][i];
        intermediateVelocity_z.at(i) = velocities_z[0][i];
    }

    // adjust the initial velocities of all bodies
    adjustVelocities(simulationData);

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
    computeAccelerations(queue, masses, positions_x[0], positions_y[0], positions_z[0],
                         acceleration_x, acceleration_y, acceleration_z);

    // compute energy of the initial step.
    computeEnergy(simulationData, currentStep);

    // store norm of all accelerations
    storeAccelerations(currentStep, acceleration_x, acceleration_y, acceleration_z);

    // continue with next simulation step
    time += dt;
    timeSinceLastVisualization += dt;
    currentStep += 1;

    // vectors for intermediate values during the time integration
    std::vector<double> vx_k1_2(numberOfBodies);
    std::vector<double> vy_k1_2(numberOfBodies);
    std::vector<double> vz_k1_2(numberOfBodies);


    // start of simulation
    while (time < t_end) {

        bool visualizeCurrentStep = (std::abs(timeSinceLastVisualization - visualizationStepWidth) < 0.000001);

        // leapfrog integration part 1 updates the position
        for (std::size_t i = 0; i < numberOfBodies; ++i) {
            vx_k1_2.at(i) = intermediateVelocity_x.at(i) + acceleration_x.at(i) * (dt / 2.0);
            vy_k1_2.at(i) = intermediateVelocity_y.at(i) + acceleration_y.at(i) * (dt / 2.0);
            vz_k1_2.at(i) = intermediateVelocity_z.at(i) + acceleration_z.at(i) * (dt / 2.0);

            // get new position
            double px_k_1 = intermediatePosition_x.at(i) + vx_k1_2.at(i) * dt;
            double py_k_1 = intermediatePosition_y.at(i) + vy_k1_2.at(i) * dt;
            double pz_k_1 = intermediatePosition_z.at(i) + vz_k1_2.at(i) * dt;

            // save positions for further computations
            intermediatePosition_x.at(i) = px_k_1;
            intermediatePosition_y.at(i) = py_k_1;
            intermediatePosition_z.at(i) = pz_k_1;

            if (visualizeCurrentStep) {
                // store the values for the ParaView output
                positions_x[currentStep].push_back(px_k_1);
                positions_y[currentStep].push_back(py_k_1);
                positions_z[currentStep].push_back(pz_k_1);
            }
        }

        // update the acceleration values, only depends on new position of bodies
        computeAccelerations(queue, masses, intermediatePosition_x, intermediatePosition_y, intermediatePosition_z,
                             acceleration_x, acceleration_y, acceleration_z);

        if (visualizeCurrentStep) {
            // store norm of all accelerations for the output
            storeAccelerations(currentStep, acceleration_x, acceleration_y, acceleration_z);
        }

        // leapfrog integration part 2, update velocities based on the new computed acceleration
        for (std::size_t i = 0; i < numberOfBodies; ++i) {
            // calculate new velocity
            double vx_k_1 = vx_k1_2.at(i) + acceleration_x.at(i) * (dt / 2.0);
            double vy_k_1 = vy_k1_2.at(i) + acceleration_y.at(i) * (dt / 2.0);
            double vz_k_1 = vz_k1_2.at(i) + acceleration_z.at(i) * (dt / 2.0);

            // save velocities for further computation
            intermediateVelocity_x.at(i) = vx_k_1;
            intermediateVelocity_y.at(i) = vy_k_1;
            intermediateVelocity_z.at(i) = vz_k_1;

            if (visualizeCurrentStep) {
                // store the values for the ParaView output
                velocities_x[currentStep].push_back(vx_k_1);
                velocities_y[currentStep].push_back(vy_k_1);
                velocities_z[currentStep].push_back(vz_k_1);
            }
        }

        if (visualizeCurrentStep) {
            // compute and store energy values of the system in the current time step
            computeEnergy(simulationData, currentStep);

            // reset time for next visualization time step
            currentStep += 1;
            timeSinceLastVisualization = 0.0;
        }

        // continue with next simulation step
        time += dt;
        timeSinceLastVisualization += dt;
    }

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

void NaiveAlgorithm::computeEnergy(const SimulationData &simulationData, size_t currentStep) {
    double E_kin = 0;
    double E_pot = 0;
    double E_total;

    // compute kinetic energy in this time step
    for (int i = 0; i < numberOfBodies; ++i) {
        double v = velocities_x[currentStep].at(i) * velocities_x[currentStep].at(i) +
                        velocities_y[currentStep].at(i) * velocities_y[currentStep].at(i) +
                        velocities_z[currentStep].at(i) * velocities_z[currentStep].at(i);
        E_kin += 0.5 * simulationData.mass[i] * v;
    }

    // compute potential energy in this time step
    for (int j = 0; j < numberOfBodies; ++j) {
        for (int i = 0; i < j; ++i) {
            double r_x = positions_x[currentStep][j] - positions_x[currentStep][i];
            double r_y = positions_y[currentStep][j] - positions_y[currentStep][i];
            double r_z = positions_z[currentStep][j] - positions_z[currentStep][i];

            double r = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

            E_pot += G * simulationData.mass[i] * simulationData.mass[j] / r;
        }
    }

    E_pot *= -1;
    // total energy
    E_total = E_kin + E_pot;

    totalEnergy[currentStep] = E_total;
    potentialEnergy[currentStep] = E_pot;
    kineticEnergy[currentStep] = E_kin;
    virialEquilibrium[currentStep] = (2.0 * E_kin) / std::abs(E_pot);
}

void NaiveAlgorithm::computeAccelerations(queue &queue, buffer<double> &masses, std::vector<double> &currentPositions_x,
                                          std::vector<double> &currentPositions_y, std::vector<double> &currentPositions_z,
                                          std::vector<double> &acceleration_x, std::vector<double> &acceleration_y,
                                          std::vector<double> &acceleration_z) {
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


}

void NaiveAlgorithm::storeAccelerations(std::size_t currentStep, std::vector<double> &acceleration_x,
                                        std::vector<double> &acceleration_y, std::vector<double> &acceleration_z) {

    for (std::size_t i = 0; i < numberOfBodies; ++i) {
        double accelerationNorm = acceleration_x[i] * acceleration_x[i] +
                                  acceleration_y[i] * acceleration_y[i] +
                                  acceleration_z[i] * acceleration_z[i];
        acceleration[currentStep].push_back(std::sqrt(accelerationNorm));
    }

}