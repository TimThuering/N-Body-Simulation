#ifndef N_BODY_SIMULATION_NBODYALGORITHM_HPP
#define N_BODY_SIMULATION_NBODYALGORITHM_HPP

#include <string>
#include <map>
#include <cmath>
#include "SimulationData.hpp"
#include "TimeMeasurement.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

/*
 * Class from which all classes containing an implementation of an n-body algorithm are derived from.
 *
 * Implements a function for generating output files which can be used in ParaView for visualization of the simulation.
 */
class nBodyAlgorithm {
public:
    std::string description; // The name of the algorithm

    std::string outputDirectory;
    double dt; // delta t determines the time step width used for the simulation
    double t_end; // determines when the simulation will stop
    double visualizationStepWidth; // determines the step width used for the visualization (i.e. the ParaView output)
    double G;

    TimeMeasurement timer;

    // maps simulation step to computed position values
    std::map<std::size_t, std::vector<double>> positions_x;
    std::map<std::size_t, std::vector<double>> positions_y;
    std::map<std::size_t, std::vector<double>> positions_z;

    // maps simulation step to computed velocity values
    std::map<std::size_t, std::vector<double>> velocities_x;
    std::map<std::size_t, std::vector<double>> velocities_y;
    std::map<std::size_t, std::vector<double>> velocities_z;

    // maps simulation step to computed acceleration values
    std::map<std::size_t, std::vector<double>> acceleration;

    std::map<std::size_t, double> kineticEnergy;
    std::map<std::size_t, double> potentialEnergy;
    std::map<std::size_t, double> totalEnergy;
    std::map<std::size_t, double> virialEquilibrium;

    nBodyAlgorithm(double dt, double t_end, double visualizationStepWidth, std::string &outputDirectory) {
        this->dt = dt;
        this->t_end = t_end;
        this->visualizationStepWidth = visualizationStepWidth;
        this->outputDirectory = outputDirectory;

        // Gravitational constant G
        this->G = 6.67428 * std::pow(10, -11);
        double meter_AU = 1.0 / (1.49597870691 * std::pow(10, 11));
        double second_Days = 1.0 / 86400;

        // scale G appropriately
        G = G * (std::pow(meter_AU, 3) / std::pow(second_Days, 2));
    }

    /*
    * Function that starts the actual computation of the n-body simulation
    * Has to be implemented in the subclass and will execute the corresponding algorithm.
    */
    virtual void startSimulation(const SimulationData &simulationData) = 0;

    /*
     * Generates all files necessary for the ParaView visualization
     */
    void generateParaViewOutput(const SimulationData &simulationData);

    /*
     * Generates .csv files containing the positions of all bodies in the last time step
     */
    void outputLastState(const std::string &path);

    /*
     * Computes the kinetic, potential and total energy and the virial equilibrium in the current step and stores
     * the computed values for the ParaView output.
     *
     * The function makes use of SYCL for parallel computation.
     * The 7 buffers masses, current_position_{x,y,z} and velocities_{x,y,z} contain the respective values of each body in the current time step.
     */
    void computeEnergy(queue &queue, buffer<double> &masses, std::size_t currentStep,
                       buffer<double> &currentPositions_x,
                       buffer<double> &currentPositions_y, buffer<double>
                       &currentPositions_z,
                       buffer<double> &currentVelocities_x, buffer<double>
                       &currentVelocities_y,
                       buffer<double> &currentVelocities_z);

    /*
     * Compute the norm of the x,y and z component of the acceleration of each body in the current time step and saves
     * the values.
     */
    void storeAccelerations(std::size_t currentStep, buffer<double> &acceleration_x,
                            buffer<double> &acceleration_y, buffer<double> &acceleration_z);

    /*
     * This function adjusts the initial velocity values and will be used once before the first simulation step
     */
    void adjustVelocities(const SimulationData &simulationData);

private:
    // Helper functions for generating the ParaView output:

    /*
     * Helper function which writes the file header to the .vtp-File.
     */
    void writeFileHeader(size_t i, std::ofstream &vtpFile);

    /*
     * Helper function which writes the positions of all bodies to the .vtp-File.
     */
    void writePositions(size_t i, std::ofstream &vtpFile);

    /*
     * Helper function which writes the IDs of all bodies to the .vtp-File.
     * The ID is represented through the index of the body in the simulation data arrays.
     */
    void writeIDs(size_t i, std::ofstream &vtpFile);

    /*
     * Helper function which writes the accelerations of all bodies to the .vtp-File.
     */
    void writeAccelerations(size_t i, std::ofstream &vtpFile);

    /*
     * Helper function which writes the velocities of all bodies to the .vtp-File.
     */
    void writeVelocities(size_t i, std::ofstream &vtpFile);

    /*
     * Helper function which writes the masses of all bodies to the .vtp-File.
     */
    void writeMasses(const SimulationData &simulationData, std::ofstream &vtpFile) const;

    /*
     * Helper function which writes the names of all bodies to the .vtp-File.
     * Each character of the name will be converted to its corresponding ASCII representation.
     * The characters in ASCII format will be separated by one space
     * Furthermore, a "0" is added to the end of each ASCII representation of a name.
     */
    void writeNamesASCII(const SimulationData &simulationData, std::ofstream &vtpFile) const;

    /*
     * Helper function which writes the body class of each body to the .vtp-File.
     */
    void writeBodyClasses(const SimulationData &simulationData, std::ofstream &vtpFile);

    /*
    * Helper function which writes the offsets to the .vtp-File.
    */
    void writeOffsets(size_t i, std::ofstream &vtpFile);

    /*
    * Helper function which writes the connectives to the .vtp-File.
    */
    void writeConnectivity(size_t i, std::ofstream &vtpFile);

    /*
    * Helper function which writes the kinetic energy in each simulation time step to the .vtp-File.
    */
    void writeKineticEnergy(size_t i, std::ofstream &vtpFile);

    /*
    * Helper function which writes the potential energy in each simulation time step to the .vtp-File.
    */
    void writePotentialEnergy(size_t i, std::ofstream &vtpFile);

    /*
    * Helper function which writes the total energy in each simulation time step to the .vtp-File.
    */
    void writeTotalEnergy(size_t i, std::ofstream &vtpFile);

    /*
    * Helper function which writes the virial equilibrium in each simulation time step to the .vtp-File.
    */
    void writeVirialEquilibrium(size_t i, std::ofstream &vtpFile);
};

#endif //N_BODY_SIMULATION_NBODYALGORITHM_HPP
