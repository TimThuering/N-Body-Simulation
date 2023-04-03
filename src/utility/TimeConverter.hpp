#ifndef N_BODY_SIMULATION_TIMECONVERTER_HPP
#define N_BODY_SIMULATION_TIMECONVERTER_HPP


#include <string>

class TimeConverter {
public:
    /*
     * This function converts a time string of the format <double><h|d|m|y> into a double value representing the time in
     * earth days.
     * Example: 1h will be converted into 0.0416667; 1y will be converted into 365.25.
     */
    static double convertToEarthDays(std::string &time);
};


#endif //N_BODY_SIMULATION_TIMECONVERTER_HPP
