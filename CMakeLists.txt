cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)

option(USE_DPCPP "Use DPC++" ON)

# DPC++
if(${USE_DPCPP} )
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
endif()


project(N_Body_Simulation)
include_directories(src/utility)
include_directories(src/simulationData)
include_directories(src/simulationBackend)
include_directories(src/simulationBackend/BarnesHutTreeAlgorithms)

# uncomment for debugging
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-limit-debug-info")


add_executable(N_Body_Simulation
        src/main.cpp
        src/simulationData/InputParser.cpp
        src/simulationBackend/NaiveAlgorithm.cpp
        src/simulationBackend/nBodyAlgorithm.cpp
        src/utility/TimeConverter.cpp
        src/simulationBackend/BarnesHutAlgorithm.cpp src/simulationBackend/BarnesHutAlgorithm.cpp src/simulationBackend/BarnesHutAlgorithm.hpp
        src/utility/TimeMeasurement.cpp src/utility/TimeMeasurement.hpp src/simulationBackend/BarnesHutTreeAlgorithms/BarnesHutOctree.cpp src/simulationBackend/BarnesHutTreeAlgorithms/BarnesHutOctree.hpp src/utility/Configuration.hpp src/utility/Configuration.cpp src/simulationBackend/BarnesHutTreeAlgorithms/ParallelOctreeTopDownSynchronized.cpp src/simulationBackend/BarnesHutTreeAlgorithms/ParallelOctreeTopDownSynchronized.hpp)


# hipSYCL
if (NOT ${USE_DPCPP})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/lib/llvm-10/include/openmp")
    find_package(hipSYCL CONFIG REQUIRED)
    find_package(OpenMP REQUIRED)
    add_sycl_to_target(TARGET N_Body_Simulation SOURCES
            src/main.cpp
            src/simulationData/InputParser.cpp
            src/simulationBackend/NaiveAlgorithm.cpp
            src/simulationBackend/nBodyAlgorithm.cpp
            src/utility/TimeConverter.cpp
            src/simulationBackend/BarnesHutAlgorithm.cpp src/simulationBackend/BarnesHutAlgorithm.cpp src/simulationBackend/BarnesHutAlgorithm.hpp
            src/utility/TimeMeasurement.cpp src/utility/TimeMeasurement.hpp src/simulationBackend/BarnesHutTreeAlgorithms/BarnesHutOctree.cpp src/simulationBackend/BarnesHutTreeAlgorithms/BarnesHutOctree.hpp src/utility/Configuration.hpp src/utility/Configuration.cpp src/simulationBackend/BarnesHutTreeAlgorithms/ParallelOctreeTopDownSynchronized.cpp src/simulationBackend/BarnesHutTreeAlgorithms/ParallelOctreeTopDownSynchronized.hpp)
endif ()




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
        src/simulationData/InputParser.cpp
        tests/TimeConverterTest.cpp
        src/utility/TimeConverter.cpp)

add_executable(BarnesHutTest
        tests/BarnesHutTest.cpp
        src/simulationBackend/nBodyAlgorithm.cpp
        src/simulationBackend/BarnesHutAlgorithm.cpp
        src/utility/TimeMeasurement.cpp src/simulationBackend/BarnesHutTreeAlgorithms/BarnesHutOctree.cpp src/simulationBackend/BarnesHutTreeAlgorithms/BarnesHutOctree.hpp src/utility/Configuration.hpp src/utility/Configuration.cpp)




enable_testing()
include(GoogleTest)
gtest_discover_tests(InputParserTest BarnesHutTest)
target_link_libraries(InputParserTest GTest::gtest_main)
target_link_libraries(BarnesHutTest GTest::gtest_main)

FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.0.0
)
FetchContent_MakeAvailable(cxxopts)
add_dependencies(N_Body_Simulation cxxopts)
target_include_directories(N_Body_Simulation PUBLIC ${cxxopts_SOURCE_DIR}/include)



