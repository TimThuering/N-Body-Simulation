#ifndef N_BODY_SIMULATION_PREPROCESSING_DETAIL_UTILITY_HPP_
#define N_BODY_SIMULATION_PREPROCESSING_DETAIL_UTILITY_HPP_

#include <string>      // std::string
#include <string_view> // std::string_view
#include <vector>      // std::vector

namespace preprocess::detail {

/**
 * @brief Convert the given @p val to a double. If the string is empty, returns
 * 0.0.
 *
 * @param[in] val the string to convert
 * @return the double value of the string @p val or 0.0 if @p val is empty
 * (`[[nodiscard]]`)
 */
[[nodiscard]] double convert_to_double_if_given(const std::string &val);

/**
 * @brief Convert the @p angle from degrees to radian.
 *
 * @param[in] angle the angle to convert to radian
 * @return the angle in radian (`[[nodiscard]]`)
 */
[[nodiscard]] double degree_to_radian(double angle);

/**
 * @brief Return a new string where all occurrences of the character @p exclude
 * from the @p str are removed.
 *
 * @param[in] str the string to remove the character from
 * @param[in] exclude the character to remove
 * @return the string @p str where all occurrences of @p exclude are removed
 * (`[[nodiscard]]`)
 */
[[nodiscard]] std::string remove_char_from_string(const std::string &str,
                                                  char exclude);

/**
 * @brief Split the string @p str by the delimiter @p delim.
 * @details If the input string @p str is empty, returns an empty vector.
 *
 * @param[in] str the string to split
 * @param[in] delim the delimiter character
 * @return the vector containing the split substrings of @p str
 * (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<std::string> split(std::string_view str, char delim);

} // namespace preprocess::detail

#endif // N_BODY_SIMULATION_PREPROCESSING_DETAIL_UTILITY_HPP_