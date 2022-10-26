#include "InputParser.h"
#include "Body.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

void InputParser::parse_input(std::string &path) {
    std::ifstream fileInputStream(path);
    std::string dataString;

    // skip first line
    std::getline(fileInputStream, dataString);

    // read the input file line by line and create the body objects
    while (std::getline(fileInputStream, dataString)) {
        // split the line into tokens
        auto tokens = InputParser::splitString(dataString);

        std::array<double, 3> position{{getDoubleScientific(tokens[4]),
                                        getDoubleScientific(tokens[5]),
                                        getDoubleScientific(tokens[6])}};

        std::array<double, 3> velocity{{getDoubleScientific(tokens[7]),
                                        getDoubleScientific(tokens[8]),
                                        getDoubleScientific(tokens[9])}};
        // create the body
        struct Body body;
        body.id = std::stol(tokens[0]);
        body.name = tokens[1];
        body.body_class = tokens[2];
        body.mass = getDoubleScientific(tokens[3]);
        body.position = position;
        body.velocity = velocity;


        allBodies.push_back(body);
        std::cout << allBodies.size() << std::endl;
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

double InputParser::getDoubleScientific(const std::string &string) {
    std::stringstream scientificNotation(string);
    double doubleFromSci;
    scientificNotation >> doubleFromSci;
    return doubleFromSci;
}