#include "io.hpp"

#include "coordinates.hpp" // preprocess::{keplerian_orbital_elements, cartesian_state_vector}
#include "detail/utility.hpp" // preprocess::detail{split, remove_char_from_string, convert_to_double_if_given, degree_to_radian}
#include "orbit_class.hpp"    // nbody::orbit_class_from_string

#include <algorithm> // std::sort, std::unique
#include <cstddef>   // std::size_t
#include <fstream>   // std::ifstream
#include <iostream>  // std::cerr, std::endl
#include <map>       // std::map
#include <set>       // std::set
#include <string>    // std::string

namespace preprocess {

std::vector<keplerian_orbital_elements>
read_orbital_elements_from_files(const std::vector<std::string> &file_names) {
  std::vector<keplerian_orbital_elements> parsed_result;

  // parse all files in the list one after another
  for (const std::string &file_name : file_names) {
    const std::vector<keplerian_orbital_elements> result_from_file =
        read_orbital_elements_from_file(file_name);
    parsed_result.reserve(parsed_result.size() + result_from_file.size());
    parsed_result.insert(parsed_result.end(), result_from_file.begin(),
                         result_from_file.end());
  }

  // remove duplicate entries
  remove_duplicate_bodies(parsed_result);

  return parsed_result;
}

std::vector<keplerian_orbital_elements>
read_orbital_elements_from_file(const std::string &file_name) {
  std::vector<keplerian_orbital_elements> parsed_result;
  std::size_t count_invalid_entries = 0;

  // output some information
  std::cout << "  parsing file \"" << file_name << "\" ...\n";

  std::ifstream in{file_name};
  if (in.fail()) {
    throw std::runtime_error{"Error: can't open or read file: " + file_name};
  }

  std::string line;

  // first line contains header information
  std::getline(in, line, '\n');
  const std::vector<std::string> column_names = detail::split(line, ',');

  // create map from column name to column index
  std::map<std::string, std::size_t> column_mapping;
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    column_mapping[detail::remove_char_from_string(column_names[i], '"')] = i;
  }

  // read data
  while (std::getline(in, line, '\n')) {
    const std::vector<std::string> values = detail::split(line, ',');

    keplerian_orbital_elements c;
    // semi - major axis in AU
    c.a = detail::convert_to_double_if_given(values[column_mapping.at("a")]);
    // eccentricity
    c.e = detail::convert_to_double_if_given(values[column_mapping.at("e")]);
    // argument of periapsis in rad
    c.w = detail::degree_to_radian(
        detail::convert_to_double_if_given(values[column_mapping.at("w")]));
    // longitude of ascending node in rad
    c.om = detail::degree_to_radian(
        detail::convert_to_double_if_given(values[column_mapping.at("om")]));
    // inclination in rad
    c.i = detail::degree_to_radian(
        detail::convert_to_double_if_given(values[column_mapping.at("i")]));
    // mean anomaly in rad
    c.ma = detail::degree_to_radian(
        detail::convert_to_double_if_given(values[column_mapping.at("ma")]));
    // epoch
    c.epoch =
        detail::convert_to_double_if_given(values[column_mapping.at("epoch")]);

    // absolute magnitude
    c.H = detail::convert_to_double_if_given(values[column_mapping.at("H")]);
    // geometric albedo
    c.albedo =
        detail::convert_to_double_if_given(values[column_mapping.at("albedo")]);
    // diameter in m
    c.diameter = detail::convert_to_double_if_given(
                     values[column_mapping.at("diameter")]) *
                 1000.0;
    // mass in kg
    if (column_mapping.count("mass") == 1) {
      // mass column exists
      c.mass =
          detail::convert_to_double_if_given(values[column_mapping.at("mass")]);
    }

    // name
    c.name = values[column_mapping.at("name")];
    // orbit class
    c.orb_class = orbit_class_from_string(values[column_mapping.at("class")]);
    // central body
    if (column_mapping.count("central_body") == 1) {
      // central_body column exists
      c.central_body = values[column_mapping.at("central_body")];
    } else {
      // assume sun as central body
      c.central_body = "Sun";
    }

    // if none of absolute magnitude, geometric albedo, diameter, or mass is
    // given, ignore this body since the mass can't be approximated
    if (c.H == 0.0 && c.albedo == 0.0 && c.diameter == 0.0 && c.mass == 0.0) {
      // can't estimate mass -> discard body
      // std::cerr << line << std::endl;
      ++count_invalid_entries;
      continue;
    }

    parsed_result.push_back(std::move(c));
  }

  // remove duplicate entries
  remove_duplicate_bodies(parsed_result);

  std::cout << "    -> found " << parsed_result.size() << " valid bodies and "
            << count_invalid_entries
            << " bodies for which no mass could be estimated\n";

  return parsed_result;
}

void write_state_vectors_to_file(
    const std::string &file_name,
    const std::vector<cartesian_state_vector> &state_vectors) {
  std::ofstream out{file_name};
  if (out.fail()) {
    throw std::runtime_error{"Error: can't create or open file: " + file_name};
  }

  // output header information
  out << "id,name,class,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z" << std::endl;
  // output Cartesian state vectors one after another
  for (std::size_t i = 0; i < state_vectors.size(); ++i) {
    const cartesian_state_vector &csv = state_vectors[i];
    out << i << ',' << csv.name << ',' << orbit_class_to_string(csv.orb_class)
        << ',' << csv.mass << ',' << csv.x << ',' << csv.y << ',' << csv.z
        << ',' << csv.vx << ',' << csv.vy << ',' << csv.vz << '\n';
  }
}

void remove_duplicate_bodies(
    std::vector<keplerian_orbital_elements> &orbital_elements) {
  // sort all Keplerian orbital elements by name
  std::sort(
      orbital_elements.begin(), orbital_elements.end(),
      [](const keplerian_orbital_elements &a,
         const keplerian_orbital_elements &b) { return a.name < b.name; });
  // sort all Keplerian orbital elements with the same name
  orbital_elements.erase(
      std::unique(orbital_elements.begin(), orbital_elements.end(),
                  [](const keplerian_orbital_elements &a,
                     const keplerian_orbital_elements &b) {
                    return a.name == b.name && !a.name.empty() &&
                           !b.name.empty();
                  }),
      orbital_elements.end());
}

} // namespace preprocess