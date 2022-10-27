#include "InputParser.hpp"
#include "SimulationData.hpp"

#include <fstream>
#include <iostream>

void InputParser::parse_input(std::string &path, SimulationData &simulationData) {
    std::ifstream fileInputStream(path);
    std::string dataString;

    // skip first line
    std::getline(fileInputStream, dataString);

    // read the input file line by line and create the body objects
    while (std::getline(fileInputStream, dataString)) {
        // split the line into tokens
        auto tokens = InputParser::splitString(dataString);

        // store all the body data in the simulationData struct
        simulationData.names.push_back(tokens[1]);
        simulationData.body_classes.push_back(tokens[2]);
        simulationData.mass.push_back(std::stod(tokens[3]));

        simulationData.positions_x.push_back(std::stod(tokens[4]));
        simulationData.positions_y.push_back(std::stod(tokens[5]));
        simulationData.positions_z.push_back(std::stod(tokens[6]));

        simulationData.velocities_x.push_back(std::stod(tokens[7]));
        simulationData.velocities_y.push_back(std::stod(tokens[8]));
        simulationData.velocities_z.push_back(std::stod(tokens[9]));
    }
}

std::vector<std::string> InputParser::splitString(std::string string) {
    std::vector<std::string> result;

    int lastSplit = 0;
    for (auto i = 0; i < string.size(); ++i) {
        if (string[i] == ',') {
            // if the delimiter character (",") has been reached, add the first token to the result and update the
            // position off the last split
            result.push_back(string.substr(lastSplit, i - lastSplit));
            lastSplit = i + 1;
        }
    }

    // add the last token to the result
    result.push_back(string.substr(lastSplit, (string.size()) - lastSplit));

    return result;
}