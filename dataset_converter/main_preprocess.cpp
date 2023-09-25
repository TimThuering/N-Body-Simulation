#include "preprocess.hpp"

#include "orbit_class.hpp" // nbody::orbit_class, nbody::orbit_class_to_full_name, nbody::orbit_class_to_string

#include "cxxopts.hpp" // cxxopts::Options, cxxopts::ParseResult

#include <cstddef>    // std::size_t
#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <filesystem> // std::filesystem::{*}
#include <iostream>   // std::cerr, std::endl
#include <map>        // std::map
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <utility>    // std::make_pair
#include <vector>     // std::vector

int main(int argc, char **argv) {
  int exit_code = EXIT_SUCCESS;

  try {
    // parse command line parameter
    cxxopts::Options options(argv[0], "n-body simulation");
    options.set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("o,out", "the filename to store the processed state vectors", cxxopts::value<std::string>()->default_value("state_vectors.csv"))
            ("d,diameter", "the minimal diameter for a main-belt asteroid to be used (in m)", cxxopts::value<double>()->default_value("0.0"))
            ("filenames", "the files or directories used as source for the orbital elements", cxxopts::value<std::vector<std::string>>())
            ("h,help", "print this helper message", cxxopts::value<bool>());
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
      options.parse_positional({"filenames"});
      result = options.parse(argc, argv);
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n' << options.help() << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // print help message and exit
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      std::exit(EXIT_SUCCESS);
    }

    std::vector<std::string> filenames;
    for (const std::string &name :
         result["filenames"].as<std::vector<std::string>>()) {
      // if "name" is a directory, add all regular files in the directory
      std::filesystem::path p{name};
      if (std::filesystem::is_directory(p)) {
        // iterate over directory
        for (const auto &dir_entry :
             std::filesystem::recursive_directory_iterator(p)) {
          // only regular csv files should be used
          if (dir_entry.is_regular_file() &&
              dir_entry.path().extension().string() == ".csv") {
            filenames.emplace_back(dir_entry.path().string());
          }
        }
      } else if (std::filesystem::is_regular_file(p)) {
        // add explicitly given file
        filenames.emplace_back(name);
      } else {
        // invalid value given
        throw std::runtime_error{"Can't parse given string: " + name};
      }
    }

    // check if at least one file could be found
    if (filenames.empty()) {
      throw std::runtime_error{"Error: no file could be found with the given "
                               "command line parameter!"};
    }

    // read the Keplerian orbital elements and remove duplicates
    std::cout << "Reading Keplerian orbital elements...\n";
    std::vector<preprocess::keplerian_orbital_elements> orbital_elements =
        preprocess::read_orbital_elements_from_files(filenames);

    // exclude all main-belt asteroids if a diameter less than the requested one
    preprocess::filter_orbital_elements(
        orbital_elements, [=](const preprocess::keplerian_orbital_elements &k) {
          return k.orb_class != preprocess::orbit_class::MBA ||
                 k.diameter >= result["diameter"].as<double>();
        });

    // approximate values if not given
    std::cout << "Approximate values...\n";
    preprocess::approximate_values(orbital_elements);

    // convert Keplerian orbital elements to Cartesian state vectors
    std::cout << "Converting Keplerian orbital elements to Cartesian state "
                 "vectors...\n";
    std::vector<preprocess::cartesian_state_vector> state_vectors =
        preprocess::orbital_elements_to_state_vectors(orbital_elements);

    // write the Cartesian state vectors to file
    const std::string output_filename = result["out"].as<std::string>();
    std::cout << "Writing cartesian state vectors to \"" << output_filename
              << "\" ...\n";
    preprocess::write_state_vectors_to_file(output_filename, state_vectors);

    // final message with some statistics
    std::cout << "Created simulation file with " << state_vectors.size()
              << " bodies:\n";
    std::map<preprocess::orbit_class, std::size_t> statistics;
    // count how often each orbit class exists
    for (const preprocess::cartesian_state_vector &c : state_vectors) {
      if (statistics.count(c.orb_class) == 0) {
        // does not already exist in mapping -> create new entry with value 1
        statistics.insert(std::make_pair(c.orb_class, 1));
      } else {
        ++statistics[c.orb_class];
      }
    }
    // print statistics
    for (const auto &[key, val] : statistics) {
      std::cout << '\t' << preprocess::orbit_class_to_full_name(key) << " ("
                << preprocess::orbit_class_to_string(key) << "): " << val
                << '\n';
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit_code = EXIT_FAILURE;
  }
  return exit_code;
}
