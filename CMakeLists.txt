cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-limit-debug-info")

project(N_Body_Simulation)
include_directories(src/simulationData)
set(CMAKE_CXX_STANDARD 17)


add_executable(N_Body_Simulation
        src/main.cpp
        src/simulationData/Body.cpp
        src/simulationData/InputParser.cpp)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(InputParserTest
        tests/InputParserTest.cpp
        src/simulationData/InputParser.cpp)

enable_testing()
include(GoogleTest)
gtest_discover_tests(InputParserTest)
target_link_libraries(InputParserTest GTest::gtest_main)

FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.0.0
)
FetchContent_MakeAvailable(cxxopts)
add_dependencies(N_Body_Simulation cxxopts)
target_include_directories(N_Body_Simulation PUBLIC ${cxxopts_SOURCE_DIR}/include)



