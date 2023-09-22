#include "orbit_class.hpp" // nbody::orbit_class

#include <stdexcept>   // std::runtime_error
#include <string>      // std::string
#include <string_view> // std::string_view
#include <type_traits> // std::underlying_type_t

namespace preprocess {

orbit_class orbit_class_from_string(const std::string &name) {
  if (name == "STA") {
    return orbit_class::STA;
  } else if (name == "PLA") {
    return orbit_class::PLA;
  } else if (name == "DWA") {
    return orbit_class::DWA;
  } else if (name == "SAT") {
    return orbit_class::SAT;
  } else if (name == "AMO") {
    return orbit_class::AMO;
  } else if (name == "APO") {
    return orbit_class::APO;
  } else if (name == "AST") {
    return orbit_class::AST;
  } else if (name == "ATE") {
    return orbit_class::ATE;
  } else if (name == "CEN") {
    return orbit_class::CEN;
  } else if (name == "HYA") {
    return orbit_class::HYA;
  } else if (name == "IEO") {
    return orbit_class::IEO;
  } else if (name == "MCA") {
    return orbit_class::MCA;
  } else if (name == "IMB") {
    return orbit_class::IMB;
  } else if (name == "MBA") {
    return orbit_class::MBA;
  } else if (name == "OMB") {
    return orbit_class::OMB;
  } else if (name == "PAA") {
    return orbit_class::PAA;
  } else if (name == "TJN") {
    return orbit_class::TJN;
  } else if (name == "TNO") {
    return orbit_class::TNO;
  } else {
    throw std::runtime_error{"Illegal orbit class: " + name};
  }
}

std::string_view orbit_class_to_string(const orbit_class oc) {
  switch (oc) {
  case orbit_class::STA:
    return "STA";
  case orbit_class::PLA:
    return "PLA";
  case orbit_class::DWA:
    return "DWA";
  case orbit_class::SAT:
    return "SAT";
  case orbit_class::AMO:
    return "AMO";
  case orbit_class::APO:
    return "APO";
  case orbit_class::AST:
    return "AST";
  case orbit_class::ATE:
    return "ATE";
  case orbit_class::CEN:
    return "CEN";
  case orbit_class::HYA:
    return "HYA";
  case orbit_class::IEO:
    return "IEO";
  case orbit_class::MCA:
    return "MCA";
  case orbit_class::IMB:
    return "IMB";
  case orbit_class::MBA:
    return "MBA";
  case orbit_class::OMB:
    return "OMB";
  case orbit_class::PAA:
    return "PAA";
  case orbit_class::TJN:
    return "TJN";
  case orbit_class::TNO:
    return "TNO";
  }
}

std::string_view orbit_class_to_full_name(const orbit_class oc) {
  switch (oc) {
  case orbit_class::STA:
    return "Star/Sun/Sol";
  case orbit_class::PLA:
    return "Planets";
  case orbit_class::DWA:
    return "Dwarf Planets";
  case orbit_class::SAT:
    return "Satellites/Moons";
  case orbit_class::AMO:
    return "Amor";
  case orbit_class::APO:
    return "Apollo";
  case orbit_class::AST:
    return "Asteroid";
  case orbit_class::ATE:
    return "Aten";
  case orbit_class::CEN:
    return "Centaur";
  case orbit_class::HYA:
    return "Hyperbolic Asteroid";
  case orbit_class::IEO:
    return "Interior Earth Object";
  case orbit_class::MCA:
    return "Mars-crossing Asteroid";
  case orbit_class::IMB:
    return "Inner Main-belt Asteroid";
  case orbit_class::MBA:
    return "Main-belt Asteroid";
  case orbit_class::OMB:
    return "Outer Main-belt Asteroid";
  case orbit_class::PAA:
    return "Parabolic Asteroid";
  case orbit_class::TJN:
    return "Jupiter Trojan";
  case orbit_class::TNO:
    return "TransNeptunian Object";
  }
}

std::underlying_type_t<orbit_class>
orbit_class_to_underlying(const orbit_class oc) {
  return static_cast<std::underlying_type_t<orbit_class>>(oc);
}

} // namespace preprocess