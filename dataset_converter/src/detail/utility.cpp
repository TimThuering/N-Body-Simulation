#include "detail/utility.hpp"

#include <algorithm>   // std::copy_if
#include <cmath>       // M_PI
#include <iterator>    // std::back_inserter
#include <string>      // std::string, std::stod
#include <string_view> // std::string_view
#include <vector>      // std::vector

namespace preprocess::detail {

double convert_to_double_if_given(const std::string &val) {
  return val.empty() ? 0.0 : std::stod(val);
}

double degree_to_radian(const double degree) { return degree * M_PI / 180.0; }

std::string remove_char_from_string(const std::string &str,
                                    const char exclude) {
  std::string res;
  std::copy_if(str.cbegin(), str.cend(), std::back_inserter(res),
               [exclude](const char c) { return c != exclude; });
  return res;
}

std::vector<std::string> split(const std::string_view str, const char delim) {
  std::vector<std::string> splitted;

  // if the input string is empty, return an empty string
  if (str.empty()) {
    return splitted;
  }

  std::string_view::size_type pos = 0;
  std::string_view::size_type next = 0;
  // split string
  while (next != std::string_view::npos) {
    next = str.find_first_of(delim, pos);
    splitted.emplace_back(next == std::string_view::npos
                              ? str.substr(pos)
                              : str.substr(pos, next - pos));
    pos = next + 1;
  }
  return splitted;
}

} // namespace preprocess::detail