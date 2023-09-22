#ifndef N_BODY_SIMULATION_PREPROCESSING_CONVERSION_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_CONVERSION_HPP_

#include "coordinates.hpp" // preprocess::keplerian_orbital_elements, preprocess::cartesian_state_vector

#include <vector> // std::vector

namespace preprocess {

/**
 * @brief Converts all Keplerian orbital elements in @p orbital_elements to
 * their Cartesian state vectors equivalent.
 *
 * @param[in] orbital_elements the Keplerian orbital elements to convert to
 * Cartesian state vectors
 * @exception std::runtime_error thrown if the central body of a body could not
 * be found
 * @attention Will always add the Sun/Sol to the output vector!
 * @return the Cartesian state vectors corresponding to @p orbital_elements
 * (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<cartesian_state_vector>
orbital_elements_to_state_vectors(
    const std::vector<keplerian_orbital_elements> &orbital_elements);

/**
 * @brief Convert the Keplerian orbital element @p kep to a Cartesian state
 * vector.
 * @brief Loosely based on:
 * https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
 *
 * @param[in] kep the Keplerian orbital element to convert
 * @param[in] central_body the body which @p kep orbits (only interesting for
 * moons)
 * @param[in] t the Julian Date epoch of the reference body (here: the Sun/Sol)
 * @param[in] t_0 the Julia Date epoch of @p kep
 * @return the Cartesian state vector (`[[nodiscard]]`)
 */
[[nodiscard]] cartesian_state_vector
orbital_element_to_state_vector(const keplerian_orbital_elements &kep,
                                const cartesian_state_vector &central_body,
                                double t, double t_0);

} // namespace preprocess

#endif // N_BODY_SIMULATION_PREPROCESSING_CONVERSION_HPP_