#ifndef N_BODY_SIMULATION_PREPROCESSING_COORDINATES_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_COORDINATES_HPP_

#include "orbit_class.hpp"  // orbit_class

#include <iosfwd>  // forward declaration of std::ostream
#include <string>  // std::string

namespace preprocess {

/**
 * @brief Struct encapsulating the necessary information for a Cartesian state vector together with additional information.
 * @details This information include:
 * - the body's position in three dimension in AU
 * - the body's velocity in three dimensions in AU/d
 * - the body's mass in kg
 * - the body's IAU name
 * - the body's orbit class
 */
struct cartesian_state_vector {
    /// The position in x-dimension in AU.
    double x{};
    /// The position in y-dimension in AU.
    double y{};
    /// The position in z-dimension in AU.
    double z{};
    /// The velocity in x-dimension in AU/d.
    double vx{};
    /// The velocity in y-dimension in AU/d.
    double vy{};
    /// The velocity in z-dimension in AU/d.
    double vz{};

    /// The mass in kg of the body represented by the cartesian state vectors.
    double mass{};
    /// The name of the body represented by the cartesian state vectors.
    std::string name{};
    /// The orbit class of the body represented by the cartesian state vectors.
    orbit_class orb_class{};
};

/**
 * @brief Output-stream operator for a Cartesian state vector.
 *
 * @param[in,out] out the output-stream
 * @param[in] csv the Cartesian state vector to output
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cartesian_state_vector &csv);

/**
 * @brief Struct encapsulating the necessary information for a Keplerian orbital element together with additional information.
 * @details This information include:
 * - the Keplerian orbital elements: semi-major axis, eccentricity, argument of periapsis, longitude of the ascending node, inclination, mean anomaly, and epoch
 * - the body's absolute magnitude
 *   - the body's geometric albedo
 * - the body's diameter in km
 * - the body's mass in kg
 * - the body's IAU name
 * - the body's orbit class
 * - the name of the body this body orbits
 */
struct keplerian_orbital_elements {
    /// The semi-major axis in AU.
    double a{};
    /// The eccentricity.
    double e{};
    /// The argument of periapsis in radian.
    double w{};
    /// The longitude of the ascending node in radian.
    double om{};
    /// The inclination in radian.
    double i{};
    /// The mean anomaly \f$ M_0 = M(t_0) \f$ at epoch \f$ t_0 \f$
    double ma{};
    /// The epoch as JD (Julian Date) for which the above Keplerian orbital elements are valid.
    double epoch{};

    /// The absolute magnitude.
    double H{};
    /// The (estimated) geometric albedo.
    double albedo{};
    /// The (estimated) diameter in m.
    double diameter{};
    /// The (estimated) mass in kg.
    double mass{};

    /// The name of the body represented by the cartesian state vectors.
    std::string name{};
    /// The orbit class of the body represented by the cartesian state vectors.
    orbit_class orb_class{};
    /// The name of the body this body orbits (for moons: the IAU name of the planet it orbits, otherwise the Sun/Sol).
    std::string central_body{};
};

/**
 * @brief Output-stream operator for a Keplerian orbital element.
 *
 * @param[in,out] out the output-stream
 * @param[in] koe the Keplerian orbital element to output
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const keplerian_orbital_elements &koe);

}  // namespace preprocess

#endif  // N_BODY_SIMULATION_PREPROCESSING_COORDINATES_HPP_