#include <gtest/gtest.h>
#include "InputParser.hpp"

TEST(SplitStringTest, BasicCase) {
    auto result = InputParser::splitString("5,Name,Class,5000,0.5,-0.3,0.4,0.06,0.1,0.2");
    EXPECT_EQ(result.at(0), "5");
    EXPECT_EQ(result.at(1), "Name");
    EXPECT_EQ(result.at(2), "Class");
    EXPECT_EQ(result.at(3), "5000");
    EXPECT_EQ(result.at(4), "0.5");
    EXPECT_EQ(result.at(5), "-0.3");
    EXPECT_EQ(result.at(6), "0.4");
    EXPECT_EQ(result.at(7), "0.06");
    EXPECT_EQ(result.at(8), "0.1");
    EXPECT_EQ(result.at(9), "0.2");
    EXPECT_EQ(result.size(), 10);
}

TEST(SplitStringTest, EmptyEntry) {
    auto result = InputParser::splitString("5,,Class,5000,0.5,-0.3,0.4,0.06,0.1,0.2");
    EXPECT_EQ(result.at(0), "5");
    EXPECT_EQ(result.at(1), "");
    EXPECT_EQ(result.at(2), "Class");
    EXPECT_EQ(result.at(3), "5000");
    EXPECT_EQ(result.at(4), "0.5");
    EXPECT_EQ(result.at(5), "-0.3");
    EXPECT_EQ(result.at(6), "0.4");
    EXPECT_EQ(result.at(7), "0.06");
    EXPECT_EQ(result.at(8), "0.1");
    EXPECT_EQ(result.at(9), "0.2");
    EXPECT_EQ(result.size(), 10);
}