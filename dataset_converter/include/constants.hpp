#ifndef N_BODY_SIMULATION_PREPROCESSING_CONSTANTS_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_CONSTANTS_HPP_

namespace preprocess {

namespace detail {

/// The gravitational constant using the SI units: \f$ \frac{m^3}{kg \cdot s^2}
/// \f$.
constexpr double gravitational_constant_si = 6.67428e-11;
/// Conversion from 1 AU (Astronomical Unit) to meter.
constexpr double au_in_m = 1.49597870691e11;
/// Conversion from 1 day to seconds.
constexpr double day_in_s = 86400;

} // namespace detail

/// The gravitational constant using AU and days instead of meter and seconds:
/// \f$ \frac{AU^3}{kg \cdot d^2} \f$.
constexpr double gravitational_constant =
    (detail::day_in_s * detail::day_in_s) *
    (detail::gravitational_constant_si /
     (detail::au_in_m * detail::au_in_m * detail::au_in_m));

/// The softening factor used in the force calculation to prevent collisions
/// between bodies.
constexpr double softening = 1e-11;

} // namespace nbody

#endif // N_BODY_SIMULATION_PREPROCESSING_CONSTANTS_HPP_
