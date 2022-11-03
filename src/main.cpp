#include "SimulationData.hpp"
#include "InputParser.hpp"
#include "NaiveAlgorithm.hpp"
#include "TimeConverter.hpp"
#include <cxxopts.hpp>

int main(int argc, char *argv[]) {
    cxxopts::Options arguments("N-Body-Simulation");

    // add the programm arguments
    arguments.add_options()
            ("file", "Path to a .csv file containing the data for the simulation",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("dt", "Width of the time step for the simulation",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("t_end", "t_end specifies the time until the system will be simulated",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("vs", "The time step width of the visualization",
             cxxopts::value<std::string>());

    arguments.add_options()
            ("vs_dir", "The top-level output directory for the ParaView output files",
             cxxopts::value<std::string>());


    auto options = arguments.parse(argc, argv);

    // input and output paths
    std::string path = options["file"].as<std::string>();
    std::string outputDirectoryPath = options["vs_dir"].as<std::string>();

    // time values for the simulation
    std::string dt_input = options["dt"].as<std::string>();
    std::string t_end_input = options["t_end"].as<std::string>();
    std::string vs_input = options["vs"].as<std::string>();

    // convert input to double values in earth days
    double dt = TimeConverter::convertToEarthDays(dt_input);
    double t_end = TimeConverter::convertToEarthDays(t_end_input);
    double visualizationStepWidth = TimeConverter::convertToEarthDays(vs_input);

    // Storage for simulation the data
    SimulationData simulationData;


    // parse the csv file containing the simulation data
    InputParser::parse_input(path, simulationData);

    NaiveAlgorithm algorithm(dt, t_end, visualizationStepWidth, outputDirectoryPath, simulationData.mass.size());


    algorithm.startSimulation(simulationData);
    algorithm.generateParaViewOutput(simulationData);
}
