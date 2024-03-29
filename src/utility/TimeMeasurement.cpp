#include "TimeMeasurement.hpp"
#include <iostream>
#include <fstream>
#include "Configuration.hpp"
#include <string>

void TimeMeasurement::addTimingSequence(const std::string &name) {
    std::vector<double> timeSequence;
    times[name] = timeSequence;
}

void TimeMeasurement::addTimeToSequence(const std::string &sequenceName, double time) {
    times[sequenceName].push_back(time);
}

void TimeMeasurement::exportJSON(const std::string &path) {


    // generate the JSON file containing all the measured timings
    std::ofstream jsonFile(path);
    jsonFile << "{ \n";
    jsonFile << "  " << "\"algorithm\": " << "\"" << algorithmType << "\",\n";
    jsonFile << "  " << "\"device\": " << "\"" << device << "\",\n";
    if (algorithmType == "Naive Algorithm") {
        jsonFile << "  " << "\"block size\": " << configuration::naive_algorithm::blockSize << ",\n";
        jsonFile << "  " << "\"optimization stage\": " << configuration::naive_algorithm::optimization_stage << ",\n";
    } else {
        jsonFile << "  " << "\"theta\": " << configuration::barnes_hut_algorithm::theta << ",\n";
        jsonFile << "  " << "\"work-group size acceleration\": " << configuration::barnes_hut_algorithm::workGroupSize << ",\n";
        jsonFile << "  " << "\"work-items AABB\": " << configuration::barnes_hut_algorithm::AABBWorkItemCount << ",\n";
        jsonFile << "  " << "\"work-items octree\": " << configuration::barnes_hut_algorithm::octreeWorkItemCount
                 << ",\n";
        jsonFile << "  " << "\"work-items center of mass\": "
                 << configuration::barnes_hut_algorithm::centerOfMassWorkItemCount << ",\n";
#ifndef OCTREE_TOP_DOWN_SYNC
        jsonFile << "  " << "\"work-items top octree\": " << configuration::barnes_hut_algorithm::octreeTopWorkItemCount
                 << ",\n";
        jsonFile << "  " << "\"max build-level top octree\": " << configuration::barnes_hut_algorithm::maxBuildLevel
                 << ",\n";
#endif
        jsonFile << "  " << "\"bodies sorted\": " << configuration::barnes_hut_algorithm::sortBodies
                 << ",\n";
    }
    jsonFile << "  " << "\"body count\": " << bodyCount;


    for (auto &[sequenceName, timeSequence]: times) {
        if (!timeSequence.empty()) {
            jsonFile << ",\n";

            jsonFile << "  \"" << sequenceName << "\"" << ": " << "[";
            for (int t = 0; t < timeSequence.size() - 1; ++t) {
                jsonFile << timeSequence[t] << ", ";
            }
            jsonFile << timeSequence[timeSequence.size() - 1] << "]";
        }

    }
    jsonFile << "\n}";

    jsonFile.close();
}

void TimeMeasurement::setProperties(std::string &algorithm, d_type::int_t &bodyCountArg, std::string &deviceArg) {
    this->algorithmType = algorithm;
    this->bodyCount = bodyCountArg;
    this->device = deviceArg;
}
