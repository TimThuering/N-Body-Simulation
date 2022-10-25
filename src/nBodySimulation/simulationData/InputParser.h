#ifndef N_BODY_SIMULATION_INPUTPARSER_H
#define N_BODY_SIMULATION_INPUTPARSER_H

#include <string>
#include <array>
#include <vector>

/*
 * Class that offers functions for parsing a CSV file containing the description of all bodies used for the simulation.
 */
class InputParser {
public:
    /*
     * This function parses the file containing the description of all bodies used for the simulation located at "path"
     *
     * The file must have the following layout:
     * id,name,class,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z
     *
     * The first line of the file will be ignored. The data is expected to begin at the second line.
     *
     */
    static void parse_input(std::string &path);

    /*
     * This helper function splits a string into tokens. It uses "," as the delimiter. This function is by the parse_input
     * function for splitting one line of the CSV-file into tokens.
     *
     * It returns a vector of strings, where each entry represents one token.
     */
    static std::vector<std::string> splitString(std::string string);

    /*
     * This helper function converts a string with a number written in scientific notation into a double.
     * E.g.: "1e+2" will result into 100
     *
     * The function enables parse_input to parse CSV-files containing data written in scientific notation.
     */
    static double getDoubleScientific(const std::string &string);
};


#endif //N_BODY_SIMULATION_INPUTPARSER_H
