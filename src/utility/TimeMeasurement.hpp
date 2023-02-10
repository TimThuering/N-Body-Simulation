#ifndef N_BODY_SIMULATION_TIMEMEASUREMENT_HPP
#define N_BODY_SIMULATION_TIMEMEASUREMENT_HPP

#include <map>
#include <vector>

class TimeMeasurement {
    std::map<std::string, std::vector<long>> times;

public:
    std::string algorithmType;
    std::size_t bodyCount;
    std::string device;
    /*
     * Adds a new key value pair to the map times with name as key.
     */
    void addTimingSequence(const std::string& name);

    /*
     * Adds a time entry to a sequence previously created with the function addTimingSequence.
     */
    void addTimeToSequence(const std::string& sequenceName, long time);

    void exportJSON(const std::string& path);

    void setProperties(std::string &algorithm, std::size_t &bodyCount, std::string &device);
};


#endif //N_BODY_SIMULATION_TIMEMEASUREMENT_HPP
