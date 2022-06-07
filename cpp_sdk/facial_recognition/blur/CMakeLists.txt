cmake_minimum_required(VERSION 3.12)
project(
        cpp_sample_apps_fr_blur
        DESCRIPTION "Trueface C++ SDK sample app, facial recognition, Blur"
        LANGUAGES CXX
)

# Check if token has been defined as environment variable
if(DEFINED ENV{TRUEFACE_TOKEN})
    add_definitions(-DTRUEFACE_TOKEN="$ENV{TRUEFACE_TOKEN}")
else()
    # TODO: Either define token as environment variable, or need to manually enter below
    add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -ffast-math")
if (UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRUEFACE_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../../trueface_sdk")

include_directories(${TRUEFACE_SDK_DIR}/include)
link_directories(${TRUEFACE_SDK_DIR}/lib)

if(DEFINED ENV{OpenCV_PATH})
    find_package( OpenCV REQUIRED PATHS $ENV{OpenCV_PATH})
else()
    find_package( OpenCV REQUIRED)
endif()

add_executable(cpp_sample_app_fr_blur src/main.cpp)
target_link_libraries(cpp_sample_app_fr_blur tf ${OpenCV_LIBS})