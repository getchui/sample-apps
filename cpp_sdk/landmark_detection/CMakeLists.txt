cmake_minimum_required(VERSION 3.12)
project(
        cpp_sample_apps_landmark_detection
        DESCRIPTION "Trueface C++ SDK sample app, landmark detection"
        LANGUAGES CXX
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -ffast-math -fopenmp")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRUEFACE_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../trueface_sdk")

include_directories(${TRUEFACE_SDK_DIR})
link_directories(${TRUEFACE_SDK_DIR})

find_package( OpenCV REQUIRED )

add_executable(cpp_sample_app_landmark_detection src/main.cpp)
target_link_libraries(cpp_sample_app_landmark_detection tf ${OpenCV_LIBS})