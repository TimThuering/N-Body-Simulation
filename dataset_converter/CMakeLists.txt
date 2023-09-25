
add_executable(preprocess
        main_preprocess.cpp
        src/detail/utility.cpp
        src/approximation.cpp
        src/conversion.cpp
        src/coordinates.cpp
        src/io.cpp
        src/orbit_class.cpp
)

add_dependencies(preprocess cxxopts)
target_include_directories(preprocess PUBLIC ${cxxopts_SOURCE_DIR}/include)
target_include_directories(preprocess PUBLIC include/)