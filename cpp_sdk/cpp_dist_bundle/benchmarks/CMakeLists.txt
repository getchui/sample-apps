cmake_minimum_required(VERSION 3.15)
project(
        trueface_sdk_benchmarks
        DESCRIPTION "Speed benchmarks for Trueface C++ SDK"
        LANGUAGES CXX
        )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -Wno-deprecated-declarations")
# Require OpenMP for Linux and windows builds
if (UNIX AND NOT APPLE OR WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRUEFACE_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../trueface_sdk")
include_directories(${TRUEFACE_SDK_DIR}/include)
link_directories(${TRUEFACE_SDK_DIR}/lib)

add_executable(run_benchmarks benchmark.cpp)
add_executable(run_benchmarks_1N_identification benchmark_1N_identification.cpp)

IF (WIN32)
    target_link_libraries(run_benchmarks libtf)
    target_link_libraries(run_benchmarks_1N_identification libtf)
ELSE()
    target_link_libraries(run_benchmarks tf)
    target_link_libraries(run_benchmarks_1N_identification tf)
ENDIF()