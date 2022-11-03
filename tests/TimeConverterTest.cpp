#include <gtest/gtest.h>
#include "TimeConverter.hpp"

TEST(TimeConversionTests, convertHour1) {
    std::string input = "1h";
    double result = TimeConverter::convertToEarthDays(input);
    EXPECT_EQ(result, 0.0416667);
}

TEST(TimeConversionTests, convertHour2) {
    std::string input = "10h";
    double result = TimeConverter::convertToEarthDays(input);
    EXPECT_EQ(result, 0.416667);
}

TEST(TimeConversionTests, convertMonth) {
    std::string input = "1m";
    double result = TimeConverter::convertToEarthDays(input);
    EXPECT_EQ(result, 30.4167);
}

TEST(TimeConversionTests, convertYear) {
    std::string input = "1y";
    double result = TimeConverter::convertToEarthDays(input);
    EXPECT_EQ(result, 365.25);
}

TEST(TimeConversionTests, convertYear3) {
    std::string input = "0.5y";
    double result = TimeConverter::convertToEarthDays(input);
    EXPECT_EQ(result, 182.625);
}

TEST(TimeConversionTests, invalidDescriptor) {
    std::string input = "0.5s";
    EXPECT_THROW(TimeConverter::convertToEarthDays(input), std::invalid_argument);
}

TEST(TimeConversionTests, noDescriptor) {
    std::string input = "0.5";
    EXPECT_THROW(TimeConverter::convertToEarthDays(input), std::invalid_argument);
}

TEST(TimeConversionTests, invalidFormat) {
    std::string input = "1y 5m";
    EXPECT_THROW(TimeConverter::convertToEarthDays(input), std::invalid_argument);
}