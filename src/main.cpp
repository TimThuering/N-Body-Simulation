#include "SimulationData.hpp"
#include "InputParser.hpp"
#include "NaiveAlgorithm.hpp"
#include <cxxopts.hpp>

int main(int argc, char *argv[]) {
    cxxopts::Options arguments("N-Body-Simulation");

    // add "path" as program argument
    arguments.add_options()
            ("file", "Path to a .csv file containing the data for the simulation", cxxopts::value<std::string>());

    auto result = arguments.parse(argc, argv);
    std::string path = result["file"].as<std::string>();

    // Storage for simulation the data
    SimulationData simulationData;

    NaiveAlgorithm algorithm;

    // parse the csv file containing the simulation data
    InputParser::parse_input(path,simulationData);

    algorithm.startSimulation(simulationData);
    algorithm.generateParaViewOutput(simulationData);
}
