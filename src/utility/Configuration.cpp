#include "Configuration.hpp"
#include <cmath>

// initialize with default values
d_type::int_t configuration::numberOfBodies = 0;
double configuration::epsilon2 = std::pow(10, -22);
bool configuration::compute_energy = false;
bool configuration::use_GPUs = true;

int configuration::naive_algorithm::blockSize = 64;
int configuration::naive_algorithm::optimization_stage = 2;

d_type::int_t configuration::barnes_hut_algorithm::storageSizeParameter = 0;
int configuration::barnes_hut_algorithm::AABBWorkItemCount = 100;
int configuration::barnes_hut_algorithm::octreeWorkItemCount = 640;
int configuration::barnes_hut_algorithm::octreeTopWorkItemCount = 640;
int configuration::barnes_hut_algorithm::centerOfMassWorkItemCount = 640;
double configuration::barnes_hut_algorithm::theta = 1.05;
int configuration::barnes_hut_algorithm::maxBuildLevel = 7;
d_type::int_t configuration::barnes_hut_algorithm::stackSize = 0;
bool configuration::barnes_hut_algorithm::sortBodies = true;
int configuration::barnes_hut_algorithm::workGroupSize = 64;

void configuration::initializeConfigValues(d_type::int_t bodyCount, int storageSizeParam, int stackSizeParam) {
    configuration::numberOfBodies = bodyCount;
    configuration::barnes_hut_algorithm::storageSizeParameter = storageSizeParam * numberOfBodies;
    // increase the stack size for small body amounts of bodies to avoid problems with small theta values
    if (bodyCount < 15000) {
        configuration::barnes_hut_algorithm::stackSize = stackSizeParam * (d_type::int_t) std::ceil(std::log2(bodyCount)) + 500;
    } else {
        configuration::barnes_hut_algorithm::stackSize = stackSizeParam * (d_type::int_t) std::ceil(std::log2(bodyCount));
    }
}

void configuration::setBlockSize(int blockSize) {
    configuration::naive_algorithm::blockSize = blockSize;
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

void configuration::setSortBodies(bool sort_bodies) {
    configuration::barnes_hut_algorithm::sortBodies = sort_bodies;
}

void configuration::setDeviceGPU(bool useGPU) {
    configuration::use_GPUs = useGPU;
}

void configuration::setWorkGroupSizeBarnesHut(int workGroupSize) {
    configuration::barnes_hut_algorithm::workGroupSize = workGroupSize;
}

void configuration::setCenterOfMassWorkItemCount(int workItemCount) {
    configuration::barnes_hut_algorithm::centerOfMassWorkItemCount = workItemCount;
}

void configuration::setOptimizationStage(int stage) {
    configuration::naive_algorithm::optimization_stage = stage;
};