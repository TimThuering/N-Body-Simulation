#include "TimeMeasurement.hpp"
#include <iostream>
#include <fstream>

void TimeMeasurement::addTimingSequence(const std::string& name) {
    std::vector<long> timeSequence;
    times[name] = timeSequence;
}

void TimeMeasurement::addTimeToSequence(const std::string& sequenceName, long time) {
    times[sequenceName].push_back(time);
}

void TimeMeasurement::exportJSON(const std::string& path) {


    // generate the JSON file containing all the measured timings
    std::ofstream jsonFile(path);
    jsonFile << "{ \n";

    bool firstLine = true;
    for (auto &[sequenceName, timeSequence] : times) {
        if (!firstLine) {
            jsonFile << ",\n";
        }
        firstLine = false;

        jsonFile <<  "\"" << sequenceName <<  "\"" << ": " <<  "[";
        for (int t = 0; t < timeSequence.size() - 1; ++t) {
            jsonFile << timeSequence[t] << ", ";
        }
        jsonFile << timeSequence[timeSequence.size() - 1] << "]";
    }
    jsonFile << "\n}";

    jsonFile.close();
}
