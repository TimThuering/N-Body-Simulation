#include "Body.h"
#include "InputParser.h"
#include <cxxopts.hpp>

int main(int argc, char *argv[]) {
    cxxopts::Options arguments("N-Body-Simulation");

    // add "path" as program argument
    arguments.add_options()
            ("path", "Path to a .csv file containing the data for the simulation", cxxopts::value<std::string>());

    auto result = arguments.parse(argc, argv);
    std::string path = result["path"].as<std::string>();

    // parse the csv file containing the simulation data
    InputParser::parse_input(path);
}
