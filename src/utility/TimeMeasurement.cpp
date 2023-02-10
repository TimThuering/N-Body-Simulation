#include "TimeMeasurement.hpp"
#include <iostream>
#include <fstream>

void TimeMeasurement::addTimingSequence(const std::string &name) {
    std::vector<long> timeSequence;
    times[name] = timeSequence;
}

void TimeMeasurement::addTimeToSequence(const std::string &sequenceName, long time) {
    times[sequenceName].push_back(time);
}

void TimeMeasurement::exportJSON(const std::string &path) {


    // generate the JSON file containing all the measured timings
    std::ofstream jsonFile(path);
    jsonFile << "{ \n";
    jsonFile << "  " << "\"algorithm\": " << "\"" << algorithmType << "\",\n";
    jsonFile << "  " << "\"device\": " << "\"" << device << "\",\n";
    jsonFile << "  " << "\"body count\": " << bodyCount;

    for (auto &[sequenceName, timeSequence]: times) {
        jsonFile << ",\n";

        jsonFile << "  \"" << sequenceName << "\"" << ": " << "[";
        for (int t = 0; t < timeSequence.size() - 1; ++t) {
            jsonFile << timeSequence[t] << ", ";
        }
        jsonFile << timeSequence[timeSequence.size() - 1] << "]";
    }
    jsonFile << "\n}";

    jsonFile.close();
}

void TimeMeasurement::setProperties(std::string &algorithm, std::size_t &bodyCountArg, std::string &deviceArg) {
    this->algorithmType = algorithm;
    this->bodyCount = bodyCountArg;
    this->device = deviceArg;
}
