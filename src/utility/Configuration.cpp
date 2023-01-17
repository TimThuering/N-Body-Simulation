#include "Configuration.hpp"
#include <cmath>

// initialize with default values
std::size_t configuration::numberOfBodies = 0;
double configuration::epsilon2 = std::pow(10, -22);

int configuration::naive_algorithm::tileSizeNaiveAlg = 64;

std::size_t configuration::barnes_hut_algorithm::storageSizeParameter = 0;
int configuration::barnes_hut_algorithm::AABBWorkItemCount = 100;
int configuration::barnes_hut_algorithm::octreeWorkItemCount = 640;
double configuration::barnes_hut_algorithm::theta = 1.05;

void configuration::loadConfiguration(std::size_t bodyCount) {
    configuration::numberOfBodies = bodyCount;

    configuration::barnes_hut_algorithm::storageSizeParameter = 16 * numberOfBodies;
}