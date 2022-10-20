#ifndef N_BODY_SIMULATION_BODY_H
#define N_BODY_SIMULATION_BODY_H

#include <string>
#include <array>
#include <vector>



/*
 * Class which represents one Body of the N-Body simulation.
 *
 */
class Body {
public:
    Body(long id, std::string &name, std::string &body_class, double mass, std::array<double, 3> &position,
         std::array<double, 3> &velocity) {
        this->id = id;
        this->name = name;
        this->body_class = body_class;
        this->mass = mass;
        this->position = position;
        this->velocity = velocity;
    };

    // Getter

    [[nodiscard]] long getId() const {
        return id;
    }

    [[nodiscard]] const std::string &getName() const {
        return name;
    }

    [[nodiscard]] const std::string &getBodyClass() const {
        return body_class;
    }

    [[nodiscard]] double getMass() const {
        return mass;
    }

    [[nodiscard]] const std::array<double, 3> &getPosition() const {
        return position;
    }

    [[nodiscard]] const std::array<double, 3> &getVelocity() const {
        return velocity;
    }

private:
    // class attributes
    long id;
    std::string name;
    std::string body_class;
    double mass;
    std::array<double, 3> position{};
    std::array<double, 3> velocity{};

};

// Vector which stores all bodies that will be used for the simulation
static std::vector<Body> allBodies;


#endif //N_BODY_SIMULATION_BODY_H
