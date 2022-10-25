#ifndef N_BODY_SIMULATION_BODY_H
#define N_BODY_SIMULATION_BODY_H

#include <string>
#include <array>
#include <vector>

/*
 * Struct which represents one Body of the N-Body simulation
 */
struct Body {
    long id;
    std::string name;
    std::string body_class;
    double mass;
    std::array<double, 3> position{};
    std::array<double, 3> velocity{};
};

// Vector which stores all bodies that will be used for the simulation
inline std::vector<Body> allBodies;


#endif //N_BODY_SIMULATION_BODY_H
