cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")

project(N_Body_Simulation)
include_directories(src/nBodySimulation/simulationData)
set(CMAKE_CXX_STANDARD 17)


add_executable(N_Body_Simulation src/main.cpp src/nBodySimulation/simulationData/Body.cpp
        src/nBodySimulation/simulationData/InputParser.cpp )

