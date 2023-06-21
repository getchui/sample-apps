cmake_minimum_required(VERSION 3.15)
project(
        trueface_sdk_benchmarks
        DESCRIPTION "Speed benchmarks for Trueface C++ SDK"
        LANGUAGES CXX
        )

# Check if token has been defined as environment variable
if(DEFINED ENV{TRUEFACE_TOKEN})
    add_definitions(-DTRUEFACE_TOKEN="$ENV{TRUEFACE_TOKEN}")
else()
    # TODO: Either define token as environment variable, or need to manually enter below
    add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast")
# Require OpenMP for Linux and windows builds
if (UNIX AND NOT APPLE OR WIN32)
    # Need to explicitly link OpenMP when using static library version of trueface sdk
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRUEFACE_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../trueface_sdk")
include_directories(${TRUEFACE_SDK_DIR}/include)
link_directories(${TRUEFACE_SDK_DIR}/lib)

add_executable(run_benchmarks benchmark.cpp)
add_executable(run_benchmarks_1N_identification benchmark_1N_identification.cpp)


# Need to explicitly link onnxruntime and libdl when using static library version of trueface sdk
IF (WIN32)
    # Don't require onnxruntime for windows
    SET(LINKER_LIBS libtf ${CMAKE_DL_LIBS})
ELSE()
    set(LINKER_LIBS tf onnxruntime ${CMAKE_DL_LIBS})
ENDIF()

target_link_libraries(run_benchmarks ${LINKER_LIBS})
target_link_libraries(run_benchmarks_1N_identification ${LINKER_LIBS})