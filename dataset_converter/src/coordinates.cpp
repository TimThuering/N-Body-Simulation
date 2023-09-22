#include "coordinates.hpp"
#include "orbit_class.hpp" // nbody::orbit_class_to_string

#include <ostream> // std::ostream

namespace preprocess {

std::ostream &operator<<(std::ostream &out, const cartesian_state_vector &csv) {
  return out << csv.name << " (" << orbit_class_to_string(csv.orb_class)
             << ", " << csv.mass << "): ([" << csv.x << ", " << csv.y << ", "
             << csv.z << "], [" << csv.vx << ", " << csv.vy << ", " << csv.vz
             << "])";
}

std::ostream &operator<<(std::ostream &out,
                         const keplerian_orbital_elements &koe) {
  return out << koe.name << " (" << orbit_class_to_string(koe.orb_class)
             << ", " << koe.central_body << ", " << koe.mass << "): (" << koe.a
             << ", " << koe.e << ", " << koe.w << ", " << koe.om << ", "
             << koe.i << ", " << koe.ma << ", " << koe.epoch << ") (" << koe.H
             << ", " << koe.albedo << ", " << koe.diameter << ")";
}

} // namespace preprocess