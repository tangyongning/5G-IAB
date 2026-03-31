cmake_minimum_required(VERSION 3.10)
project(TCCA_Simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(tcca_simulation
    main_simulation.cpp
    NetworkTopology.cpp
    FaultModel.cpp
    QoSMetrics.cpp
    TCCAFramework.cpp
    BaselineMethods.cpp
    EvaluationMetrics.cpp
)

# Include directories
target_include_directories(tcca_simulation PRIVATE ${CMAKE_SOURCE_DIR})

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tcca_simulation PRIVATE -Wall -Wextra -O2)
elseif(MSVC)
    target_compile_options(tcca_simulation PRIVATE /W4 /O2)
endif()

# Output directory
set_target_properties(tcca_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
