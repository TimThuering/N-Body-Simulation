#ifndef N_BODY_SIMULATION_PREPROCESSING_APPROXIMATION_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_APPROXIMATION_HPP_

#include "coordinates.hpp" // preprocess::keplerian_orbital_elements
#include "orbit_class.hpp" // nbody::orbit_class

#include <vector> // std::vector

namespace preprocess {

/**
 * @brief Given the orbit class, approximate the geometric albedo.
 * @details Uniformly draws the geometric albedo value from a predefined range.
 *
 * @param[in] c the orbit class used for the approximation
 * @note These are only rough approximations. Feel free to improve them!
 * @return the approximated geometric albedo (`[[nodiscard]]`)
 */
[[nodiscard]] double approximate_geometric_albedo(orbit_class c);

/**
 * @brief Approximate the diameter of an asteroid given its absolute magnitude
 * `H` and geometric albedo `p`.
 * @details Based on: https://doi.org/10.1016/j.pss.2012.03.009. Mainly applied
 * for TNOs, but we also use it for all asteroids!<br> The diameter is estimated
 * using: \f$ d = 1329p^{-0.5}10^{-0.2H} \f$
 *
 * @param[in] absolute_magnitude the absolute magnitude used for the
 * approximation
 * @param[in] geometric_albedo the geometric albedo used for the approximation
 * (may be also approximated!)
 * @return the approximated diameter in km (`[[nodiscard]]`)
 */
[[nodiscard]] double approximate_diameter(double absolute_magnitude,
                                          double geometric_albedo);

/**
 * @brief Approximate the mass of an asteroid given its diameter `d` and
 * geometric albedo `p`.
 * @details Based on:
 * https://solarsystem.nasa.gov/asteroids-comets-and-meteors/asteroids/in-depth/.
 * <br> The volume \f$ v \f$ of the asteroid is estimated by assuming a
 * **spherical shape** and the diameter: \f$ \frac{4}{3} \pi (\frac{d}{2})^2
 * \f$. <br> The density \f$ \rho \f$ is estimated by sorting the asterid into
 * one of three classes using its geometric albedo:
 *          - C-type (chondrite): most common, probably consist of clay and
 * silicate rocks; \f$ p < 0.1 \rightarrow \rho = 1.38\,\frac{g}{cm^3} \f$
 *          - S-type (stony): made up of silicate materials and nickel-iron; \f$
 * 0.1 \leq p \leq 0.2 \rightarrow \rho = 2.71\,\frac{g}{cm^3} \f$
 *          - M-type (nickel-iron): metallic; \f$ p > 0.2 \rightarrow \rho
 * = 5.32\,\frac{g}{cm^3} \f$ <br>
 *
 *          The final mass \f$ m \f$ is approximated by \f$ v \cdot \rho \f$
 * (**note**: mind the physical units!).
 *
 * @param[in] diameter the diameter (in km) used for the approximation (may be
 * also approximated!)
 * @param[in] geometric_albedo the geometric albedo used for the approximation
 * (may be also approximated!)
 * @return the approximated mass necessary for the n-body simulation
 * (`[[nodiscard]]`)
 */
[[nodiscard]] double approximate_mass(double diameter, double geometric_albedo);

/**
 * @brief Approximate all necessary values of all Keplerian orbital elements in
 * @p orbital_elements.
 *
 * @param[in, out] orbital_elements the list of all Keplerian orbital elements
 */
void approximate_values(
    std::vector<keplerian_orbital_elements> &orbital_elements);

} // namespace preprocess

#endif // N_BODY_SIMULATION_PREPROCESSING_APPROXIMATION_HPP_