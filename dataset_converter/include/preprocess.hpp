#ifndef N_BODY_SIMULATION_PREPROCESSING_PREPROCESS_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_PREPROCESS_HPP_

#include "approximation.hpp" // functions used for approximating the geometric albedo, diameter, and mass
#include "conversion.hpp" // functions used for converting Keplerian orbital elements to Cartesian state vectors
#include "coordinates.hpp" // structs representing Keplerian orbital elements and Cartesian state vectors
#include "io.hpp" // functions used for reading Keplerian orbital elements from file and writing Cartesian state vectors to file

/// The main namespace containing all public functions and classes.
/// Namespace containing all functions related to preprocessing the data, i.e.,
/// converting Keplerian orbital elements to Cartesian state vectors.
namespace preprocess {}

#endif //  N_BODY_SIMULATION_PREPROCESSING_PREPROCESS_HPP_