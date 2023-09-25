#include "conversion.hpp"

#include "constants.hpp"   // nbody::gravitational_constant
#include "orbit_class.hpp" // nbody::orbit_class

#include <algorithm> // std::find_if
#include <cmath>     // std::sqrt, std::pow, std::sin, std::cos, std::atan2
#include <cstddef>   // std::size_t
#include <stdexcept> // std::runtime_error

namespace preprocess {

cartesian_state_vector
orbital_element_to_state_vector(const keplerian_orbital_elements &kep,
                                const cartesian_state_vector &central_body,
                                double t, double t_0) {
  // mass of sun (reference point)
  constexpr double G = gravitational_constant;
  const double mu = G * central_body.mass;

  // determine the time difference in seconds
  const double delta_t = (t - t_0);
  // calculate mean anomaly at time t
  const double M_t = kep.ma + delta_t * std::sqrt(mu / std::pow(kep.a, 3));

  // solve Kepler's Equation using the Newton-Raphson Methode
  constexpr std::size_t max_iter = 30;
  double E = M_t;
  for (std::size_t iter = 0; iter < max_iter; ++iter) {
    E = E - (E - kep.e * std::sin(E) - M_t) / (1.0 - kep.e * std::cos(E));
  }
  // obtain the true anomaly
  const double v_t = 2. * std::atan2(std::sqrt(1. + kep.e) * std::sin(E / 2.0),
                                     std::sqrt(1. - kep.e) * std::cos(E / 2.0));

  // get distance to the central body
  const double r_c = kep.a * (1 - kep.e * std::cos(E));

  // obtain position and velocity vector in the orbital frame
  cartesian_state_vector o{
      r_c * std::cos(v_t),
      r_c * std::sin(v_t),
      0.0,
      (std::sqrt(mu * kep.a) / r_c) * -std::sin(E),
      (std::sqrt(mu * kep.a) / r_c) *
          (std::sqrt(1 - std::pow(kep.e, 2)) * std::cos(E)),
      0.0};

  // transform o and o_d to the inertial frame in heliocentric rectangular
  // coordinates
  cartesian_state_vector r{};
  r.x = o.x * (std::cos(kep.w) * std::cos(kep.om) -
               std::sin(kep.w) * std::cos(kep.i) * std::sin(kep.om)) -
        o.y * (std::sin(kep.w) * std::cos(kep.om) +
               std::cos(kep.w) * std::cos(kep.i) * std::sin(kep.om));
  r.y = o.x * (std::cos(kep.w) * std::sin(kep.om) +
               std::sin(kep.w) * std::cos(kep.i) * std::cos(kep.om)) +
        o.y * (std::cos(kep.w) * std::cos(kep.i) * std::cos(kep.om) -
               std::sin(kep.w) * std::sin(kep.om));
  r.z = o.x * (std::sin(kep.w) * std::sin(kep.i)) +
        o.y * (std::cos(kep.w) * std::sin(kep.i));
  r.vx = o.vx * (std::cos(kep.w) * std::cos(kep.om) -
                 std::sin(kep.w) * std::cos(kep.i) * std::sin(kep.om)) -
         o.vy * (std::sin(kep.w) * std::cos(kep.om) +
                 std::cos(kep.w) * std::cos(kep.i) * std::sin(kep.om));
  r.vy = o.vx * (std::cos(kep.w) * std::sin(kep.om) +
                 std::sin(kep.w) * std::cos(kep.i) * std::cos(kep.om)) +
         o.vy * (std::cos(kep.w) * std::cos(kep.i) * std::cos(kep.om) -
                 std::sin(kep.w) * std::sin(kep.om));
  r.vz = o.vx * (std::sin(kep.w) * std::sin(kep.i)) +
         o.vy * (std::cos(kep.w) * std::sin(kep.i));

  // set position relative to central body
  r.x += central_body.x;
  r.y += central_body.y;
  r.z += central_body.z;

  // set velocity relative to central body
  r.vx += central_body.vx;
  r.vy += central_body.vy;
  r.vz += central_body.vz;

  // set mass, name, and orbital class
  r.mass = kep.mass;
  r.name = kep.name;
  r.orb_class = kep.orb_class;

  return r;
}

std::vector<cartesian_state_vector> orbital_elements_to_state_vectors(
    const std::vector<keplerian_orbital_elements> &orbital_elements) {
  std::vector<cartesian_state_vector> state_vectors;
  // create the Sun/Sol as reference point -> can not be converted from the
  // keplerian orbital elements!
  state_vectors.emplace_back(cartesian_state_vector{
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.98847e30, "Sun", orbit_class::STA});
  // midnight January 1, 2000
  const double sol_reference_epoch = 2451544.5;

  // at first, convert all bodies orbiting the sun to cartesian coordinates
  for (const keplerian_orbital_elements &k : orbital_elements) {
    if (k.central_body == "Sun") {
      state_vectors.push_back(orbital_element_to_state_vector(
          k, state_vectors.front(), sol_reference_epoch, k.epoch));
    }
  }
  // after that, convert all bodies orbiting another celestial body (except the
  // Sun)
  for (const keplerian_orbital_elements &k : orbital_elements) {
    if (k.central_body != "Sun") {
      // get central body
      const auto it = std::find_if(state_vectors.cbegin(), state_vectors.cend(),
                                   [&](const cartesian_state_vector &c) {
                                     return c.name == k.central_body;
                                   });
      if (it == state_vectors.end()) {
        // couldn't find central body
        throw std::runtime_error{"Can't find central_body!"};
      }

      state_vectors.push_back(orbital_element_to_state_vector(
          k, *it, sol_reference_epoch, k.epoch));
    }
  }

  return state_vectors;
}

} // namespace preprocess