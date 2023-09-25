#ifndef N_BODY_SIMULATION_PREPROCESSING_IO_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_IO_HPP_

#include "coordinates.hpp" // preprocess::{keplerian_orbital_elements, cartesian_state_vector}

#include <string> // std::string
#include <vector> // std::vector

namespace preprocess {

/**
 * @brief Read all Keplerian orbital elements from the list of file denoted by
 * @p file_names.
 * @details Will also remove all duplicate entries.
 *
 * @param[in] file_names multiply file names containing the Keplerian orbital
 * elements
 * @exception std::runtime_error thrown if the file could not be opened
 * @exception std::out_of_range thrown if a required .csv header entry could not
 * be found
 * @return the Keplerian orbital elements read from multiply files with
 * duplicates removed (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<keplerian_orbital_elements>
read_orbital_elements_from_files(const std::vector<std::string> &file_names);
/**
 * @brief Read all Keplerian orbital elements from the file denoted by @p
 * file_name.
 * @details Will also remove all duplicate entries.
 *
 * @param[in] file_name the file name containing the Keplerian orbital elements
 * @exception std::runtime_error thrown if the file could not be opened
 * @exception std::out_of_range thrown if a required .csv header entry could not
 * be found
 * @return the Keplerian orbital elements read with duplicates removed
 * (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<keplerian_orbital_elements>
read_orbital_elements_from_file(const std::string &file_name);

/**
 * @brief Write the Cartesian state vectors @p state_vectors to the file @p
 * file_name.
 *
 * @param[in] file_name the name of the output file
 * @param[in] state_vectors the Cartesian state vectors to save
 * @exception std::runtime_error thrown if the file could not be created or
 * opened
 */
void write_state_vectors_to_file(
    const std::string &file_name,
    const std::vector<cartesian_state_vector> &state_vectors);

/**
 * @brief Remove all duplicate entries from the @p orbital_elements.
 * @details Two entries are expected to be the same, if they have the same
 * non-empty name.
 *
 * @param[in,out] orbital_elements the Keplerian orbital elements to remove the
 * duplicates from
 */
void remove_duplicate_bodies(
    std::vector<keplerian_orbital_elements> &orbital_elements);

/**
 * @brief Remove all Keplerian orbital elements from @p orbital_elements that do
 * not satisfy the filter @p f.
 *
 * @tparam Filter the callable type of the filter
 * @param[in,out] orbital_elements the Keplerian orbital elements to filter
 * @param[in] f the filter to apply
 */
template <typename Filter>
inline void filter_orbital_elements(
    std::vector<keplerian_orbital_elements> &orbital_elements, Filter f) {
  std::vector<keplerian_orbital_elements> temp;

  for (keplerian_orbital_elements &k : orbital_elements) {
    // move all orbital elements that satisfy the filter to the temporary vector
    if (f(k)) {
      temp.push_back(std::move(k));
    }
  }
  // override the input vector with the filtered temporary vector
  orbital_elements = temp;
}

} // namespace preprocess

#endif // N_BODY_SIMULATION_PREPROCESSING_IO_HPP_