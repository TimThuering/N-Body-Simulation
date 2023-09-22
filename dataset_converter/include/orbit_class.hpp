#ifndef N_BODY_SIMULATION_PREPROCESSING_ORBIT_CLASS_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_ORBIT_CLASS_HPP_

#include <string>      // std::string
#include <string_view> // std::string_view
#include <type_traits> // std::underlying_type_t

namespace preprocess {

/**
 * @brief Enum class for the different orbital classes.
 * @details The orbital classes are based on the small bodies orbital classes
 * (https://pdssbn.astro.umd.edu/data_other/objclass.shtml) with the exception
 * of `STA`, `PLA`, `DWA`, and `SAT` which are custom extensions for a better
 * ParaView visualization.
 */
enum class orbit_class {
  /** Star (Sol/Sun) */
  STA = 0,
  /** Planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune) */
  PLA = 1,
  /** Dwarf Planets (e.g., Ceres, Pluto, Makemake, ...) */
  DWA = 2,
  /** Satellites/Moons (e.g., Luna, Phobos, Deimos, Europa, Io, Triton, Charon,
     ...) */
  SAT = 3,
  /** Amor (near-Earth asteroid orbits similar to that of 1221 Amor) */
  AMO = 4,
  /** Apollo (near-Earth asteroid orbits which cross the Earth's orbit similar
     to that of 1862 Apollo) */
  APO = 5,
  /** Asteroid (asteroid orbit not matching any defined orbit class) */
  AST = 6,
  /** Aten (near-Earth asteroid orbits similar to that of 2062 Aten) */
  ATE = 7,
  /** Centaur (objects with orbits between Jupiter and Neptune) */
  CEN = 8,
  /** Hyperbolic Asteroid (Asteroids on hyperbolic orbits) */
  HYA = 9,
  /** Interior Earth Object (an asteroid orbit contained entirely within the
     orbit of the Earth) */
  IEO = 10,
  /** Mars-crossing Asteroid (asteroids that cross the orbit of Mars) */
  MCA = 11,
  /** Inner Main-belt Asteroid (asteroids with orbital elements constrained by
     (a < 2.0 AU; q > 1.666 AU)) */
  IMB = 12,
  /** Main-belt Asteroid (asteroids with orbital elements constrained by (2.0 AU
     < a < 3.2 AU; q > 1.666 AU)) */
  MBA = 13,
  /** Outer Main-belt Asteroid (asteroids with orbital elements constrained by
     (3.2 AU < a < 4.6 AU)) */
  OMB = 14,
  /** Parabolic Asteroid (asteroids on parabolic orbits) */
  PAA = 15,
  /** Jupiter Trojan (asteroids trapped in Jupiter's L4/L5 Lagrange points) */
  TJN = 16,
  /** TransNeptunian Object (objects with orbits outside Neptune) */
  TNO = 17
};

/**
 * @brief Convert the given std::string to the corresponding orbit class.
 *
 * @param[in] name the orbit class's abbreviation
 * @exception std::runtime_error thrown if the std::string doesn't correspond to
 * a valid orbit class abbreviation
 * @return the orbit class (`[[nodiscard]]`)
 */
[[nodiscard]] orbit_class orbit_class_from_string(const std::string &name);

/**
 * @brief Convert the given orbit class to a std::string.
 *
 * @param[in] oc the orbit class
 * @exception std::runtime_error thrown if the orbit_class @p oc isn't handled
 * in this function
 * @return the std::string representation of the orbit class @p oc
 * (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view orbit_class_to_string(orbit_class oc);

/**
 * @brief Convert the given orbit class to its full name.
 * @details If the abbreviation is sufficient, use `orbit_class_to_string`.
 *
 * @param[in] oc the orbit class
 * @exception std::runtime_error thrown if the orbit_class @p oc isn't handled
 * in this function
 * @return the full name of the orbit class @p oc (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view orbit_class_to_full_name(orbit_class oc);

/**
 * @brief Convert the orbit class to the enum value's underlying type.
 *
 * @param[in] oc the orbit class
 * @return the value converted to the underlying type of the orbit_class enum
 * (`[[nodiscard]]`)
 */
[[nodiscard]] std::underlying_type_t<orbit_class>
orbit_class_to_underlying(orbit_class oc);

} // namespace preprocess

#endif // N_BODY_SIMULATION_PREPROCESSING_ORBIT_CLASS_HPP_
