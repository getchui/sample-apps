cmake_minimum_required(VERSION 3.12)
project(
        cpp_sample_apps_fr_1_1
        DESCRIPTION "Trueface C++ SDK sample app, facial recognition, 1:1"
        LANGUAGES CXX
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -ffast-math")
if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wno-deprecated-declarations")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRUEFACE_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../../trueface_sdk")

include_directories(${TRUEFACE_SDK_DIR}/include)
link_directories(${TRUEFACE_SDK_DIR}/lib)

add_executable(cpp_sample_app_fr_1_1 src/main.cpp)
target_link_libraries(cpp_sample_app_fr_1_1 tf)