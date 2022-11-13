#include <stdexcept>
#include "TimeConverter.hpp"

double TimeConverter::convertToEarthDays(std::string &time) {
    const double hourInEarthDays = 1.0 / 24; // one hour are 0.0416667 earth days
    const double monthInEarthDays = 30.4167;  // one month are 30.4167 earth days
    const double yearInEarthDays = 365.25;    // one year are 365.25 earth days

    char descriptor = time[time.length() - 1];
    double timeNotConverted;
    double convertedTime;

    size_t charactersConverted = 0; //stores the amount of characters converted by std::stod
    size_t *charactersConverted_ptr = &charactersConverted;

    std::string prefix = time.substr(0, time.length() - 1); // the prefix of the input should be the double value

    // try to convert first part of string into double
    try {
        timeNotConverted = std::stod(prefix, charactersConverted_ptr);
    } catch (const std::exception &exception) {
        throw std::invalid_argument("Time values have to be of the format <double><h|d|m|y>");
    }

    // check if the whole prefix got converted
    if (charactersConverted != prefix.length()) {
        throw std::invalid_argument("\"Time values have to be of the format <double><h|d|m|y>\"");
    }

    // convert the prefix containing the time value according to what the descriptor (suffix) was
    switch (descriptor) {
        case 'h': // hour to earth days conversion
            convertedTime = timeNotConverted * hourInEarthDays;
            break;
        case 'd': // no conversion necessary
            convertedTime = timeNotConverted;
            break;
        case 'm': // months to earth days conversion
            convertedTime = timeNotConverted * monthInEarthDays;
            break;
        case 'y': // years to earth days conversion
            convertedTime = timeNotConverted * yearInEarthDays;
            break;
        default:
            throw std::invalid_argument("Time values have to be of the format <double><h|d|m|y>");
    }
    return convertedTime;
}