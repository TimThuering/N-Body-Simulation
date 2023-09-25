#include "approximation.hpp"

#include "orbit_class.hpp" // orbit_class

#include <cmath> // std::pow, M_PI
#include <map>   // std::map
#include <random> // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string> // std::string

namespace preprocess {

double approximate_geometric_albedo(const orbit_class c) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::map<orbit_class, std::uniform_real_distribution<double>>
      albedo_mapping = {
          // Near-Earth Asteroids (NEA): Amor, Apollo, Aten, Interior Earth
          // Object
          {orbit_class::AMO,
           std::uniform_real_distribution<double>{0.450, 0.550}}, // Amor
          {orbit_class::APO,
           std::uniform_real_distribution<double>{0.450, 0.550}}, // Apollo
          {orbit_class::AST,
           std::uniform_real_distribution<double>{0.450, 0.550}}, // Asteroid
          {orbit_class::ATE,
           std::uniform_real_distribution<double>{0.450, 0.550}}, // Aten
          {orbit_class::CEN,
           std::uniform_real_distribution<double>{
               0.040,
               0.750}}, // Centaur:
                        // https://iopscience.iop.org/article/10.3847/1538-3881/aac210/pdf
          {orbit_class::HYA,
           std::uniform_real_distribution<double>{
               0.450, 0.550}}, // Hyperbolic Asteroid
          {orbit_class::IEO,
           std::uniform_real_distribution<double>{
               0.450, 0.550}}, // Interior Earth Object
          {orbit_class::MCA,
           std::uniform_real_distribution<double>{
               0.450, 0.550}}, // Mars-crossing Asteroid
          {orbit_class::IMB,
           std::uniform_real_distribution<double>{
               0.030,
               0.103}}, // Inner Main-belt Asteroid:
                        // https://sci.esa.int/web/gaia/-/60214-gaia-s-view-of-more-than-14-000-asteroids
          {orbit_class::MBA,
           std::uniform_real_distribution<double>{
               0.097,
               0.203}}, // Main-belt Asteroid:
                        // https://sci.esa.int/web/gaia/-/60214-gaia-s-view-of-more-than-14-000-asteroids
          {orbit_class::OMB,
           std::uniform_real_distribution<double>{
               0.197,
               0.5}}, // Outer Main-belt Asteroid:
                      // https://sci.esa.int/web/gaia/-/60214-gaia-s-view-of-more-than-14-000-asteroids
          {orbit_class::PAA,
           std::uniform_real_distribution<double>{0.450,
                                                  0.550}}, // Parabolic Asteroid
          {orbit_class::TJN,
           std::uniform_real_distribution<double>{0.188,
                                                  0.124}}, // Jupiter Trojans
          {orbit_class::TNO,
           std::uniform_real_distribution<double>{
               0.022, 0.130}} // TransNeptunian Objects
      };

  return albedo_mapping.at(c)(gen);
}

double approximate_diameter(double absolute_magnitude,
                            double geometric_albedo) {
  // based on: https://doi.org/10.1016/j.pss.2012.03.009
  return 1329.0 * std::pow(geometric_albedo, -0.5) *
         std::pow(10.0, -0.2 * absolute_magnitude) * 1000.0;
}

double approximate_mass(double diameter, double geometric_albedo) {
  // based on:
  // https://solarsystem.nasa.gov/asteroids-comets-and-meteors/asteroids/in-depth/
  const double volume = 4. / 3. * M_PI * std::pow(diameter / 2.0, 3); // km^3
  double density = 0.0; // g / cm^3
  if (geometric_albedo < 0.1) {
    // C-type asteroid density
    density = 1.38;
  } else if (geometric_albedo > 0.2) {
    // M-type asteroid density
    density = 5.32;
  } else {
    // S-type asteroid density
    density = 2.71;
  }
  density = density / 1000.0 * std::pow(100, 3); // kg / m^3
  return volume * density;
}

void approximate_values(
    std::vector<keplerian_orbital_elements> &orbital_elements) {
#pragma omp parallel for default(none) shared(orbital_elements)
  for (keplerian_orbital_elements &kep : orbital_elements) {
    // approximate mass only if necessary
    if (kep.mass == 0.0) {
      // NOTE: the order must not be changed since the albedo is needed to
      // approximate the diameter approximate albedo if not provided
      if (kep.albedo == 0.0) {
        kep.albedo = approximate_geometric_albedo(kep.orb_class);
      }
      // approximate diameter if not provided
      if (kep.diameter == 0.0) {
        kep.diameter = approximate_diameter(kep.H, kep.albedo);
      }
      // approximate mass
      kep.mass = approximate_mass(kep.diameter, kep.albedo);
    }
  }
}

} // namespace preprocess