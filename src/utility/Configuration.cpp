#include "Configuration.hpp"
#include <cmath>
#include <sycl/sycl.hpp>

// initialize with default values
std::size_t configuration::numberOfBodies = 0;
double configuration::epsilon2 = std::pow(10, -22);
bool configuration::compute_energy = false;

int configuration::naive_algorithm::tileSizeNaiveAlg = 64;



std::size_t configuration::barnes_hut_algorithm::storageSizeParameter = 0;
int configuration::barnes_hut_algorithm::AABBWorkItemCount = 100;
int configuration::barnes_hut_algorithm::octreeWorkItemCount = 640;
int configuration::barnes_hut_algorithm::octreeTopWorkItemCount = 640;
double configuration::barnes_hut_algorithm::theta = 1.05;
int configuration::barnes_hut_algorithm::maxBuildLevel = 7;
std::size_t configuration::barnes_hut_algorithm::stackSize = 0;

void configuration::initializeConfigValues(std::size_t bodyCount, int storageSizeParam, int stackSizeParam) {
    configuration::numberOfBodies = bodyCount;
    configuration::barnes_hut_algorithm::storageSizeParameter = storageSizeParam * numberOfBodies;
    configuration::barnes_hut_algorithm::stackSize = stackSizeParam * (std::size_t) std::ceil(std::log2(bodyCount));

}

void configuration::setBlockSize(int blockSize) {
    configuration::naive_algorithm::tileSizeNaiveAlg = blockSize;
}

void configuration::setTheta(double theta) {
    configuration::barnes_hut_algorithm::theta = theta;
}

void configuration::setAABBWorkItemCount(int workItemCount) {
    configuration::barnes_hut_algorithm::AABBWorkItemCount = workItemCount;
}

void configuration::setOctreeWorkItemCount(int workItemCount) {
    configuration::barnes_hut_algorithm::octreeWorkItemCount = workItemCount;
}

void configuration::setOctreeTopWorkItemCount(int workItemCount) {
    configuration::barnes_hut_algorithm::octreeTopWorkItemCount = workItemCount;
}

void configuration::setMaxBuildLevel(int maxLevel) {
    configuration::barnes_hut_algorithm::maxBuildLevel = maxLevel;
}

void configuration::setEnergyComputation(bool computeEnergy) {
    configuration::compute_energy = computeEnergy;
}
