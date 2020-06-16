cmake_minimum_required(VERSION 3.12)
project(
        cpp_sample_apps
        DESCRIPTION "Trueface C++ SDK sample app"
        LANGUAGES CXX
)

add_subdirectory(facial_recognition)
add_subdirectory(landmark_detection)
add_subdirectory(object_detection)
add_subdirectory(video_analyze)
