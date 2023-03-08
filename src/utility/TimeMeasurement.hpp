#ifndef N_BODY_SIMULATION_TIMEMEASUREMENT_HPP
#define N_BODY_SIMULATION_TIMEMEASUREMENT_HPP

#include <map>
#include <vector>
#include <string>
#include "Configuration.hpp"

class TimeMeasurement {
    std::map<std::string, std::vector<double>> times;

public:
    std::string algorithmType;
    d_type::int_t bodyCount;
    std::string device;
    /*
     * Adds a new key value pair to the map times with name as key.
     */
    void addTimingSequence(const std::string& name);

    /*
     * Adds a time entry to a sequence previously created with the function addTimingSequence.
     */
    void addTimeToSequence(const std::string& sequenceName, double time);

    void exportJSON(const std::string& path);

    void setProperties(std::string &algorithm, d_type::int_t &bodyCount, std::string &device);
};


#endif //N_BODY_SIMULATION_TIMEMEASUREMENT_HPP
